import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, ModuleList
import torch.nn.functional as F
from lib.nn.models.Mytransformersparse import *


class PositionalEncodingModule(nn.Module):
    def __init__(self, size):
        super(PositionalEncodingModule, self).__init__()
        self.encoding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size)))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, -1)
        b, c, d, e = x.size()
        x = x.reshape(b, c, d * e)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        y = x * y.expand_as(x)
        y = y.reshape(b, c, d, e)
        y = y.transpose(1, -1)
        return y


class AdaCTSi(nn.Module):
    def __init__(self, d_in, group, steps_per_day=288, window=24, base_dim=96, cnn_layers=3, dropout=0., d_ff=256):
        super(AdaCTSi, self).__init__()
        self.num_nodes = d_in
        self.grouppp = group
        self.steps_per_day = steps_per_day
        self.window_size = window
        self.base_dim = base_dim
        self.embed_dim = self.base_dim
        self.node_dim = self.base_dim // 4
        self.temp_dim_tid = self.base_dim // 8
        self.temp_dim_diw = self.base_dim // 8
        self.group = 1

        print('self.base_dim', self.base_dim, 'self.group', self.group)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(self.steps_per_day, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(7, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.hidden_dim = self.embed_dim + self.node_dim + self.temp_dim_tid + self.temp_dim_diw
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.window_size, kernel_size=(1, 1), bias=True)

        self.attn_layer_s = SpatialInformerLayer(self.hidden_dim, d_ff=d_ff, dropout=dropout)
        self.cnn_layers = cnn_layers
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        # D = [1, 4, 6]
        # Extend dilation list if cnn_layers > 3, using powers of 2 or cycling
        D = [1, 4, 6]
        if self.cnn_layers > 3:
            # Simple strategy: append powers of 2 or just 1s.
            # Let's use a cycle of 1, 4, 6 for now or just 1s.
            # Actually, let's just make it [1, 4, 6, 1, 4, 6...] to be safe and consistent with original design intent if stacked.
            # Or maybe just 1s?
            # Let's use [1, 2, 4, 8...] which is more standard for TCNs if we are going deep.
            # But to preserve exact behavior for 3 layers, let's keep [1, 4, 6] as prefix.
            while len(D) < self.cnn_layers:
                D.append(2 ** (len(D) - 3))  # 1, 2, 4... after 6? No, that's weird.
                # Let's just cycle [1, 4, 6]
                # D.append(D[len(D)%3])
        # Actually, let's just use [1, 2, 4, 8, 16...] if the user changes layers?
        # No, that changes the original model.
        # Let's just ensure D is long enough.
        while len(D) < self.cnn_layers:
            D.append(2)  # Safe default dilation

        for i in range(self.cnn_layers):
            self.filter_convs.append(Conv2d(self.embed_dim, self.embed_dim, (1, 3), dilation=D[i], groups=self.group))
            self.gate_convs.append(Conv2d(self.embed_dim, self.embed_dim, (1, 3), dilation=D[i], groups=self.group))
        depth = list(range(self.cnn_layers))
        self.bn = ModuleList([BatchNorm2d(self.embed_dim) for _ in depth])

        self.start_conv = Conv2d(in_channels=4, out_channels=self.embed_dim, kernel_size=(1, 1))

    def forward(self, x, mask, adj=None, pos=None, timestamp=None, adj_label=None, complexity=None):
        # We pass complexity to internal methods
        # Use complexity from arg if provided, else use self.inference_complexity if set, else 100
        if complexity is None:
            complexity = getattr(self, 'inference_complexity', 100)

        # Determine strategy
        strategy = getattr(self, 'inference_strategy', 'default')
        limit_nodes = getattr(self, 'limit_nodes', None)

        if self.training:
            return self._forward_training(x, mask)  # Training ignores complexity (randomized)
        else:
            if strategy == 'cw':
                return self._forward_inference_codebook_mechanism(x, mask, complexity=complexity)
            elif strategy == 'arw':
                return self._forward_inference_arw_mechanism(x, mask)
            elif strategy == 'random':
                return self._forward_inference_random_mechanism(x, mask)
            elif strategy == 'drift':
                return self._forward_drift(x, mask, limit_nodes=limit_nodes)
            elif strategy == 'group':
                return self._forward_group(x, mask)
            else:  # default
                return self._forward_training(x, mask, complexity=complexity)

    def _forward_training(self, x, mask, complexity=100):
        init_x = x
        channel = x.shape[-1] // 3
        x = x[..., :channel]
        masked_x = torch.where(mask, x, torch.zeros_like(x))

        input_data_1 = masked_x.unsqueeze(-1)
        input_data_2 = init_x[..., channel:channel * 2].unsqueeze(-1)
        input_data_3 = init_x[..., channel * 2:].unsqueeze(-1)
        input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
        batch_size, _, num_nodes, _ = input_data_1.shape

        t_i_d_data = input_data[..., 1]
        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
        d_i_w_data = input_data[..., 2]
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]
        input_data = input_data.transpose(1, 3).contiguous()

        x = self.start_conv(input_data)
        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            if i == self.cnn_layers - 1:
                x = x[:, :, :, -1:]
                break
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        time_series_emb = x

        node_emb = [self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)]
        tem_emb = [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)]
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1).squeeze(-1).transpose(1, 2)

        hidden = self.attn_layer_s(hidden, complexity=complexity)
        hidden = hidden.transpose(1, 2).unsqueeze(-1)

        prediction = self.regression_layer(hidden).squeeze(-1)
        final_output = torch.where(mask, masked_x, prediction)

        return final_output, final_output

    def _forward_inference(self, x, mask):
        batch_out = []
        channel = x.shape[-1] // 3
        init_masked_x = torch.where(mask, x[..., :channel], torch.zeros_like(x[..., :channel]))
        init_x = x
        init_mask = mask

        missing_nodes_per_sample = []

        for batch_idx in range(init_x.size()[0]):
            sample_mask = mask[batch_idx]
            missing_nodes = []

            for node_idx in range(sample_mask.shape[1]):
                if torch.any(sample_mask[:, node_idx] == 0):
                    missing_nodes.append(node_idx)
            missing_nodes_per_sample.append(missing_nodes)

        for k in range(init_x.size()[0]):
            masked_x_sample = init_masked_x[k].unsqueeze(0)
            mask = init_mask[k].unsqueeze(0)
            x = init_x[k].unsqueeze(0)
            xx = x[..., :channel]
            masked_x = torch.where(mask, xx, torch.zeros_like(xx))

            input_data_1 = masked_x.unsqueeze(-1)
            input_data_2 = x[..., channel:channel * 2].unsqueeze(-1)
            input_data_3 = x[..., channel * 2:].unsqueeze(-1)
            input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
            batch_size, _, num_nodes, _ = input_data_1.shape

            t_i_d_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
            d_i_w_data = input_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]

            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            tem_emb = [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)]
            t_emb = torch.cat(tem_emb, dim=1)

            input_data = input_data.transpose(1, 3).contiguous()

            missing_list = missing_nodes_per_sample[k]

            input_data = input_data[:, :, missing_list, :]
            node_emb = node_emb[:, :, missing_list, :]
            t_emb = t_emb[:, :, missing_list, :]

            x = self.start_conv(input_data)
            for j in range(self.cnn_layers):
                residual = x
                filter = torch.tanh(self.filter_convs[j](residual))
                gate = torch.sigmoid(self.gate_convs[j](residual))
                x = filter * gate
                if j == self.cnn_layers - 1:
                    x = x[:, :, :, -1:]
                    break
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[j](x)
            time_series_emb = x

            hidden = torch.cat([time_series_emb, node_emb, t_emb], dim=1).squeeze(-1).transpose(1, 2)
            hidden = self.attn_layer_s(hidden)
            hidden = hidden.transpose(1, 2).unsqueeze(-1)

            prediction = self.regression_layer(hidden).squeeze(-1)
            masked_x_sample[:, :, missing_list] = prediction

            batch_out.append(masked_x_sample)

        prediction = torch.cat(batch_out, dim=0)
        final_output = torch.where(init_mask, init_masked_x, prediction)

        return final_output, final_output

    def _forward_drift(self, x, mask, limit_nodes=None):
        x_backup = x[..., :x.shape[-1] // 3]
        mask_backup = mask

        # 选取部分节点
        # selected_numbers = torch.randperm(self.num_nodes)[:self.num_nodes // 8]
        # selected_numbers = torch.randperm(self.num_nodes)[:self.num_nodes // 4]

        # Use limit_nodes if provided, otherwise default to 8 (or logic from user)
        # Note: If limit_nodes is provided, it typically means we want to KEEP that many nodes.
        # But the original logic used 'drift_num' to represent nodes to EXCLUDE/RESERVE (not predict).
        # "reserved_part" was the part NOT predicted.
        # "remaining_indices" was the part PREDICTED.

        # If user says limit_nodes=20 (out of 24), maybe they mean predict 20 nodes?
        # Let's assume limit_nodes means "number of nodes to PREDICT".
        # So drift_num (reserved/ignored) = total_nodes - limit_nodes.

        if limit_nodes is not None and limit_nodes > 0 and limit_nodes <= self.num_nodes:
            drift_num = self.num_nodes - limit_nodes
        else:
            # If limit_nodes not specified, default to using 75% of nodes (25% drift)
            drift_num = max(1, self.num_nodes // 4)

        reserved_part = x_backup[..., self.num_nodes - drift_num:]

        # selected_numbers = torch.randperm(self.num_nodes)[:8]
        selected_numbers = torch.arange(self.num_nodes - drift_num, self.num_nodes)
        all_indices = torch.arange(self.num_nodes)
        remaining_indices = all_indices[~torch.isin(all_indices, selected_numbers)]

        # 创建索引
        x_value_indices = remaining_indices
        x_dow_indices = remaining_indices + self.num_nodes
        x_tid_indices = remaining_indices + self.num_nodes * 2
        remaining_indices_expanded = torch.cat((x_value_indices, x_dow_indices, x_tid_indices))

        # 更新x和mask
        x = x[:, :, remaining_indices_expanded]

        mask = mask[:, :, remaining_indices]

        init_x = x
        channel = x.shape[-1] // 3
        x = x[..., :channel]
        masked_x = torch.where(mask, x, torch.zeros_like(x))

        # 准备输入数据
        input_data_1 = masked_x.unsqueeze(-1)
        input_data_2 = init_x[..., channel:channel * 2].unsqueeze(-1)
        input_data_3 = init_x[..., channel * 2:].unsqueeze(-1)
        input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
        batch_size, _, num_nodes, _ = input_data_1.shape

        # 嵌入时间信息
        t_i_d_data = input_data[..., 1]
        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
        d_i_w_data = input_data[..., 2]
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]
        input_data = input_data.transpose(1, 3).contiguous()

        # 初始卷积
        x = self.start_conv(input_data)
        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            if i == self.cnn_layers - 1:
                x = x[:, :, :, -1:]
                break
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        time_series_emb = x

        # 处理嵌入和隐藏状态
        node_emb = [
            self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)[:, :, remaining_indices,
            :]]
        tem_emb = [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)]
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1).squeeze(-1).transpose(1, 2)

        # 注意力层处理
        hidden = self.attn_layer_s(hidden)
        hidden = hidden.transpose(1, 2).unsqueeze(-1)

        # 回归层预测
        prediction = self.regression_layer(hidden).squeeze(-1)

        # 更新最终输出
        prediction = torch.where(mask, masked_x, prediction)
        output = torch.concat([prediction, reserved_part], dim=-1)

        return output, output

    def _forward_group(self, x, mask):
        init_x = x
        channel = x.shape[-1] // 3
        x = x[..., :channel]
        masked_x = torch.where(mask, x, torch.zeros_like(x))

        input_data_1 = masked_x.unsqueeze(-1)
        input_data_2 = init_x[..., channel:channel * 2].unsqueeze(-1)
        input_data_3 = init_x[..., channel * 2:].unsqueeze(-1)
        input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
        batch_size, _, num_nodes, _ = input_data_1.shape

        chunk_num = self.grouppp

        t_i_d_data = input_data[..., 1]
        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
        d_i_w_data = input_data[..., 2]
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]

        node_emb_items = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        node_emb = torch.chunk(node_emb_items, chunk_num, -2)
        tem_emb_items = torch.cat(
            [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)], dim=1)

        tem_emb = torch.chunk(tem_emb_items, chunk_num, -2)

        input_data = input_data.transpose(1, 3).contiguous()
        input_datas = torch.chunk(input_data, chunk_num, -2)
        pred_list = []
        for ii in range(chunk_num):
            x = self.start_conv(input_datas[ii])
            for i in range(self.cnn_layers):
                residual = x
                filter = torch.tanh(self.filter_convs[i](residual))
                gate = torch.sigmoid(self.gate_convs[i](residual))
                x = filter * gate

                if i == self.cnn_layers - 1:
                    x = x[:, :, :, -1:]
                    break
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[i](x)
            time_series_emb = x

            hidden = torch.cat([time_series_emb, node_emb[ii], tem_emb[ii]], dim=1).squeeze(-1).transpose(1, 2)

            hidden = self.attn_layer_s(hidden)
            hidden = hidden.transpose(1, 2).unsqueeze(-1)

            prediction = self.regression_layer(hidden).squeeze(-1)
            pred_list.append(prediction)
        final = torch.cat(pred_list, dim=-1)
        final_output = torch.where(mask, masked_x, final)

        return final_output

    def _forward_inference_random_mechanism(self, x, mask):
        batch_out = []
        channel = x.shape[-1] // 3
        init_masked_x = torch.where(mask, x[..., :channel], torch.zeros_like(x[..., :channel]))
        init_x = x
        init_mask = mask
        K = 32
        missing_nodes_per_sample = []

        for batch_idx in range(init_x.size()[0]):
            sample_mask = mask[batch_idx]
            missing_nodes = []
            not_missing_nodes = []

            for node_idx in range(sample_mask.shape[1]):
                if torch.any(sample_mask[:, node_idx] == 0):
                    missing_nodes.append(node_idx)
                else:
                    not_missing_nodes.append(node_idx)
            if len(missing_nodes) < K:
                needed = K - len(missing_nodes)
                if len(not_missing_nodes) >= needed:
                    missing_nodes.extend(random.sample(not_missing_nodes, needed))
                else:
                    missing_nodes.extend(
                        not_missing_nodes + random.choices(not_missing_nodes, k=needed - len(not_missing_nodes)))

            missing_nodes_per_sample.append(missing_nodes)

        for k in range(init_x.size()[0]):
            masked_x_sample = init_masked_x[k].unsqueeze(0)
            mask = init_mask[k].unsqueeze(0)
            x = init_x[k].unsqueeze(0)
            xx = x[..., :channel]
            masked_x = torch.where(mask, xx, torch.zeros_like(xx))

            input_data_1 = masked_x.unsqueeze(-1)
            input_data_2 = x[..., channel:channel * 2].unsqueeze(-1)
            input_data_3 = x[..., channel * 2:].unsqueeze(-1)
            input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
            batch_size, _, num_nodes, _ = input_data_1.shape

            t_i_d_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
            d_i_w_data = input_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]

            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            tem_emb = [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)]
            t_emb = torch.cat(tem_emb, dim=1)

            input_data = input_data.transpose(1, 3).contiguous()

            missing_list = missing_nodes_per_sample[k]

            input_data = input_data[:, :, missing_list, :]
            node_emb = node_emb[:, :, missing_list, :]
            t_emb = t_emb[:, :, missing_list, :]

            x = self.start_conv(input_data)
            for j in range(self.cnn_layers):
                residual = x
                filter = torch.tanh(self.filter_convs[j](residual))
                gate = torch.sigmoid(self.gate_convs[j](residual))
                x = filter * gate
                if j == self.cnn_layers - 1:
                    x = x[:, :, :, -1:]
                    break
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[j](x)
            time_series_emb = x

            hidden = torch.cat([time_series_emb, node_emb, t_emb], dim=1).squeeze(-1).transpose(1, 2)
            hidden = self.attn_layer_s(hidden)
            hidden = hidden.transpose(1, 2).unsqueeze(-1)

            prediction = self.regression_layer(hidden).squeeze(-1)
            masked_x_sample[:, :, missing_list] = prediction

            batch_out.append(masked_x_sample)

        prediction = torch.cat(batch_out, dim=0)
        final_output = torch.where(init_mask, init_masked_x, prediction)

        return final_output

    def _forward_inference_codebook_mechanism(self, x, mask, complexity=100):
        batch_out = []
        channel = x.shape[-1] // 3
        init_masked_x = torch.where(mask, x[..., :channel], torch.zeros_like(x[..., :channel]))
        init_x = x
        init_mask = mask
        K = 32
        missing_nodes_per_sample = []
        cb = self.node_emb
        for batch_idx in range(init_x.size()[0]):
            sample_mask = mask[batch_idx]
            missing_nodes = []
            not_missing_nodes = []

            for node_idx in range(sample_mask.shape[1]):
                if torch.any(sample_mask[:, node_idx] == 0):
                    missing_nodes.append(node_idx)
                else:
                    not_missing_nodes.append(node_idx)

            if len(missing_nodes) < K:
                needed = K - len(missing_nodes)
                if len(not_missing_nodes) >= needed:
                    # 计算相似性
                    missing_embs = cb[missing_nodes]
                    not_missing_embs = cb[not_missing_nodes]
                    # 计算每个非缺失节点与所有缺失节点的平均相似性
                    similarity_scores = torch.zeros(len(not_missing_nodes))
                    for i, emb in enumerate(not_missing_embs):
                        similarity = F.cosine_similarity(emb.unsqueeze(0), missing_embs, dim=1)
                        similarity_scores[i] = similarity.mean()
                    # 选择相似性最高的needed个节点
                    topk_indices = torch.topk(similarity_scores, needed).indices
                    selected_nodes = [not_missing_nodes[i] for i in topk_indices]
                    missing_nodes.extend(selected_nodes)
                else:
                    missing_nodes.extend(
                        not_missing_nodes + random.choices(not_missing_nodes, k=needed - len(not_missing_nodes)))

            missing_nodes_per_sample.append(missing_nodes)

        for k in range(init_x.size()[0]):
            masked_x_sample = init_masked_x[k].unsqueeze(0)
            mask = init_mask[k].unsqueeze(0)
            x = init_x[k].unsqueeze(0)
            xx = x[..., :channel]
            masked_x = torch.where(mask, xx, torch.zeros_like(xx))

            input_data_1 = masked_x.unsqueeze(-1)
            input_data_2 = x[..., channel:channel * 2].unsqueeze(-1)
            input_data_3 = x[..., channel * 2:].unsqueeze(-1)
            input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
            batch_size, _, num_nodes, _ = input_data_1.shape

            t_i_d_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
            d_i_w_data = input_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]

            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            tem_emb = [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)]
            t_emb = torch.cat(tem_emb, dim=1)

            input_data = input_data.transpose(1, 3).contiguous()

            missing_list = missing_nodes_per_sample[k]

            input_data = input_data[:, :, missing_list, :]
            node_emb = node_emb[:, :, missing_list, :]
            t_emb = t_emb[:, :, missing_list, :]

            x = self.start_conv(input_data)
            for j in range(self.cnn_layers):
                residual = x
                filter = torch.tanh(self.filter_convs[j](residual))
                gate = torch.sigmoid(self.gate_convs[j](residual))
                x = filter * gate
                if j == self.cnn_layers - 1:
                    x = x[:, :, :, -1:]
                    break
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[j](x)
            time_series_emb = x

            hidden = torch.cat([time_series_emb, node_emb, t_emb], dim=1).squeeze(-1).transpose(1, 2)
            hidden = self.attn_layer_s(hidden, complexity=complexity)
            hidden = hidden.transpose(1, 2).unsqueeze(-1)

            prediction = self.regression_layer(hidden).squeeze(-1)
            masked_x_sample[:, :, missing_list] = prediction

            batch_out.append(masked_x_sample)

        prediction = torch.cat(batch_out, dim=0)
        final_output = torch.where(init_mask, init_masked_x, prediction)

        return final_output, final_output

    def _forward_inference_arw_mechanism(self, x, mask):
        batch_out = []
        channel = x.shape[-1] // 3
        init_masked_x = torch.where(mask, x[..., :channel], torch.zeros_like(x[..., :channel]))
        init_x = x
        init_mask = mask
        K = 32
        missing_nodes_per_sample = []

        for batch_idx in range(init_x.size()[0]):
            sample_mask = mask[batch_idx]
            missing_nodes = []
            not_missing_nodes = []

            for node_idx in range(sample_mask.shape[1]):
                if torch.any(sample_mask[:, node_idx] == 0):
                    missing_nodes.append(node_idx)
                else:
                    not_missing_nodes.append(node_idx)
            if len(missing_nodes) < K:
                needed = K - len(missing_nodes)
                if len(not_missing_nodes) >= needed:
                    missing_nodes.extend(random.sample(not_missing_nodes, needed))
                else:
                    missing_nodes.extend(
                        not_missing_nodes + random.choices(not_missing_nodes, k=needed - len(not_missing_nodes)))

            missing_nodes_per_sample.append(missing_nodes)

        for k in range(init_x.size()[0]):
            masked_x_sample = init_masked_x[k].unsqueeze(0)
            mask = init_mask[k].unsqueeze(0)
            x = init_x[k].unsqueeze(0)
            xx = x[..., :channel]
            masked_x = torch.where(mask, xx, torch.zeros_like(xx))

            input_data_1 = masked_x.unsqueeze(-1)
            input_data_2 = x[..., channel:channel * 2].unsqueeze(-1)
            input_data_3 = x[..., channel * 2:].unsqueeze(-1)
            input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
            batch_size, _, num_nodes, _ = input_data_1.shape

            t_i_d_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.steps_per_day).long()]
            d_i_w_data = input_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).long()]

            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            tem_emb = [time_in_day_emb.transpose(1, 2).unsqueeze(-1), day_in_week_emb.transpose(1, 2).unsqueeze(-1)]
            t_emb = torch.cat(tem_emb, dim=1)

            input_data = input_data.transpose(1, 3).contiguous()

            missing_list = missing_nodes_per_sample[k]

            input_data = input_data[:, :, missing_list, :]
            node_emb = node_emb[:, :, missing_list, :]
            t_emb = t_emb[:, :, missing_list, :]

            x = self.start_conv(input_data)
            for j in range(self.cnn_layers):
                residual = x
                filter = torch.tanh(self.filter_convs[j](residual))
                gate = torch.sigmoid(self.gate_convs[j](residual))
                x = filter * gate
                if j == self.cnn_layers - 1:
                    x = x[:, :, :, -1:]
                    break
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[j](x)
            time_series_emb = x

            hidden = torch.cat([time_series_emb, node_emb, t_emb], dim=1).squeeze(-1).transpose(1, 2)
            hidden = self.attn_layer_s(hidden)
            hidden = hidden.transpose(1, 2).unsqueeze(-1)

            prediction = self.regression_layer(hidden).squeeze(-1)
            masked_x_sample[:, :, missing_list] = prediction

            batch_out.append(masked_x_sample)

        prediction = torch.cat(batch_out, dim=0)
        final_output = torch.where(init_mask, init_masked_x, prediction)

        return final_output

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--base-dim', type=int, default=96)
        parser.add_argument('--cnn-layers', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.)
        parser.add_argument('--d-ff', type=int, default=256)
        # limit-nodes is already added in run_adactsi.py but we can use it here if needed
        return parser


class AdaCTSiFullWindow(nn.Module):
    def __init__(self, d_in, group, steps_per_day=288, window=24, base_dim=96, cnn_layers=3, dropout=0., d_ff=256):
        super(AdaCTSiFullWindow, self).__init__()
        self.num_nodes = d_in
        self.grouppp = group
        self.steps_per_day = steps_per_day
        self.window_size = window
        self.base_dim = base_dim
        self.embed_dim = self.base_dim
        self.node_dim = self.base_dim // 4
        self.temp_dim_tid = self.base_dim // 8
        self.temp_dim_diw = self.base_dim // 8
        self.group = 1
        self.input_emb_dim = self.embed_dim
        self.tod_emb_dim = self.temp_dim_tid
        self.dow_emb_dim = self.temp_dim_diw
        self.spatial_emb_dim = self.node_dim
        self.adp_emb_dim = max(1, self.embed_dim // 2 - 1)
        self.model_dim = (
                self.input_emb_dim
                + self.tod_emb_dim
                + self.dow_emb_dim
                + self.spatial_emb_dim
                + self.adp_emb_dim
                + 1
        )
        self.model_dim2 = (self.model_dim - self.input_emb_dim) // 2 + self.input_emb_dim

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(self.steps_per_day, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(7, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        self.adp_emb = nn.Parameter(torch.empty(self.window_size, self.num_nodes, self.adp_emb_dim))
        nn.init.xavier_uniform_(self.adp_emb)

        self.output_proj = nn.Linear(self.window_size * self.model_dim2, self.window_size)
        self.attn_layers_s = ModuleList([
            SpatialInformerLayer(self.model_dim2, d_ff=d_ff, dropout=dropout)
        ])
        self.spatial_positional_encoding = ModuleList([
            PositionalEncodingModule((1, self.num_nodes, self.model_dim2))
        ])
        self.se = SELayer(self.model_dim - self.input_emb_dim - 1)
        self.compressed_layer = nn.Linear(
            self.model_dim - self.input_emb_dim - 1,
            (self.model_dim - self.input_emb_dim) // 2 - 1
        )
        self.cnn_layers = cnn_layers
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        D = [1, 4, 6]
        if self.cnn_layers > 3:
            while len(D) < self.cnn_layers:
                D.append(2 ** (len(D) - 3))
        while len(D) < self.cnn_layers:
            D.append(2)

        for i in range(self.cnn_layers):
            self.filter_convs.append(
                Conv2d(self.embed_dim, self.embed_dim, (1, 3), dilation=D[i], padding=(0, D[i]), groups=self.group))
        depth = list(range(self.cnn_layers))
        self.bn = ModuleList([BatchNorm2d(self.embed_dim) for _ in depth])

        self.start_conv = Conv2d(in_channels=4, out_channels=self.embed_dim, kernel_size=(1, 1))

    def forward(self, x, mask, adj=None, pos=None, timestamp=None, adj_label=None):
        if self.training:
            return self._forward_training(x, mask)
        else:
            return self._forward_training(x, mask)

    def _forward_training(self, x, mask):
        init_x = x
        channel = x.shape[-1] // 3
        x = x[..., :channel]
        masked_x = torch.where(mask, x, torch.zeros_like(x))

        input_data_1 = masked_x.unsqueeze(-1)
        input_data_2 = init_x[..., channel:channel * 2].unsqueeze(-1)
        input_data_3 = init_x[..., channel * 2:].unsqueeze(-1)
        input_data = torch.cat([input_data_1, input_data_2, input_data_3, mask.unsqueeze(-1)], dim=-1)
        batch_size, _, num_nodes, _ = input_data_1.shape

        t_i_d_data = input_data[..., 1]
        t_i_d_idx = torch.clamp((t_i_d_data * self.steps_per_day).long(), max=self.steps_per_day - 1)
        time_in_day_emb = self.time_in_day_emb[t_i_d_idx]
        d_i_w_data = input_data[..., 2]
        d_i_w_idx = torch.clamp(d_i_w_data.long(), max=6)
        day_in_week_emb = self.day_in_week_emb[d_i_w_idx]
        input_data = input_data.transpose(1, 3).contiguous()

        x = self.start_conv(input_data)
        for i in range(self.cnn_layers):
            residual = x
            x = torch.tanh(self.filter_convs[i](residual))
            if i != self.cnn_layers - 1:
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[i](x)
        time_series_emb = x

        time_len = time_series_emb.shape[-1]
        time_in_day_emb = time_in_day_emb[:, -time_len:, :, :]
        day_in_week_emb = day_in_week_emb[:, -time_len:, :, :]
        mask_slice = mask[:, -time_len:, :]

        x = time_series_emb.permute(0, 3, 2, 1).contiguous()

        spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, time_len, -1, -1)
        adp_emb = self.adp_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if adp_emb.shape[1] != time_len:
            adp_emb = adp_emb[:, -time_len:, :, :]

        total_embedding = torch.cat(
            [time_in_day_emb, day_in_week_emb, spatial_emb, adp_emb],
            dim=-1
        )
        total_embedding = self.se(total_embedding)
        total_embedding = self.compressed_layer(total_embedding)

        x = torch.cat([x, total_embedding, mask_slice.unsqueeze(-1)], dim=-1)

        for i in range(len(self.attn_layers_s)):
            temp_x = x
            x = x + self.spatial_positional_encoding[i].encoding
            x = x.reshape(batch_size * time_len, num_nodes, self.model_dim2)
            x = self.attn_layers_s[i](x)
            x = x.reshape(batch_size, time_len, num_nodes, self.model_dim2).contiguous()
            x = x + temp_x

        out = x.transpose(1, 2)
        out = out.reshape(batch_size, num_nodes, time_len * self.model_dim2)
        out = self.output_proj(out).view(batch_size, num_nodes, time_len)
        out = out.transpose(1, 2)
        final_output = out

        return final_output, final_output

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--base-dim', type=int, default=96)
        parser.add_argument('--cnn-layers', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.)
        parser.add_argument('--d-ff', type=int, default=256)
        return parser
