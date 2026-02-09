import os
import numpy as np
import pandas as pd
import torch
from .pd_dataset import PandasDataset
from ..utils.utils import infer_mask, compute_mean, geographical_distance, sample_mask, disjoint_months
from .. import datasets_path


class BaseAdapter(PandasDataset):
    def __init__(self, dataset_name, impute_nans=False, freq=None, limit_nodes=None):
        self.dataset_name = dataset_name
        self.limit_nodes = limit_nodes
        self.test_months = [3, 6, 9, 12]  # Default to standard quarterly split, can be overridden by subclasses

        # Load data based on dataset name
        df, adjs, positions, adj_label = self.load_data()

        # Handle limit_nodes for Vessel or others if needed
        if self.limit_nodes is not None and self.limit_nodes < df.shape[1]:
            print(f"Limiting nodes to {self.limit_nodes}")
            df = df.iloc[:, :self.limit_nodes]
            if adjs is not None:
                if adjs.ndim == 3:
                    adjs = adjs[:, :self.limit_nodes, :self.limit_nodes]
                else:
                    adjs = adjs[:self.limit_nodes, :self.limit_nodes]
            if positions is not None:
                if positions.ndim == 3:
                    positions = positions[:, :self.limit_nodes, :]
                else:
                    positions = positions[:self.limit_nodes, :]
            if adj_label is not None:
                if adj_label.ndim == 3:
                    adj_label = adj_label[:, :self.limit_nodes, :self.limit_nodes]
                else:
                    adj_label = adj_label[:self.limit_nodes, :self.limit_nodes]

        self.adjs = adjs
        self.positions = positions
        self.adj_label = adj_label

        # Compute mask
        mask = (~np.isnan(df.values)).astype('uint8')

        # Generate eval_mask
        # We use a simple random strategy for now as in standard adapters
        self.rng = np.random.default_rng(42)
        eval_mask = sample_mask(mask.shape, p=0.002, p_noise=0.002, min_seq=1, max_seq=1, rng=self.rng)
        self.eval_mask = (eval_mask & mask).astype('uint8')

        if impute_nans:
            df = df.fillna(df.mean())
            df = df.fillna(0)  # Fallback

        # Add Time Features (Time in Day, Day in Week)
        # 1. Time in Day: Normalized [0, 1)
        # 2. Day in Week: Integer [0, 6] (as float)

        feature_list = []
        n_nodes = df.shape[1]

        # Time in Day
        if freq == '10S':  # Vessel
            # Assuming index is datetime
            # total seconds in day = 24*3600 = 86400
            seconds_in_day = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
            time_ind = seconds_in_day / 86400.0
        else:  # Default H or similar
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")

        time_in_day = np.tile(time_ind, [1, n_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

        # Day in Week
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, n_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

        # Concatenate features
        date = np.concatenate(feature_list, axis=-1)

        # Ensure unique column names to avoid issues during resampling
        # Original df columns are likely 0..N-1 or strings.
        # We will reset columns to range(3*N) eventually, but for now let's ensure uniqueness.
        # If df.columns are integers 0..N-1:
        is_int_cols = pd.api.types.is_integer_dtype(df.columns)
        if is_int_cols:
            start_idx = df.columns.max() + 1 if len(df.columns) > 0 else 0
            cols_1 = range(start_idx, start_idx + n_nodes)
            cols_2 = range(start_idx + n_nodes, start_idx + 2 * n_nodes)
        else:
            # String columns or mixed. Use string suffixes.
            cols_1 = [f"time_{i}" for i in range(n_nodes)]
            cols_2 = [f"dow_{i}" for i in range(n_nodes)]

        new_df_1 = pd.DataFrame(date[..., 0], index=df.index, columns=cols_1)  # Time in Day
        new_df_2 = pd.DataFrame(date[..., 1], index=df.index, columns=cols_2)  # Day in Week

        final_df = pd.concat([df, new_df_1, new_df_2], axis=1)

        # Reset columns to simple range to be safe and consistent with numpy conversion
        final_df.columns = range(final_df.shape[1])

        # Do not expand mask/eval_mask.
        # AirQuality keeps mask 1x (matching only the data columns),
        # and downstream components (ImputationDataset, Scalers) handle the 1/3 slicing.

        # mask = np.concatenate([mask, mask_time, mask_time], axis=1)

        # eval_mask_time = np.zeros_like(self.eval_mask)
        # self.eval_mask = np.concatenate([self.eval_mask, eval_mask_time, eval_mask_time], axis=1)
        super().__init__(dataframe=final_df, u=None, mask=mask, name=dataset_name, freq=freq, aggr='nearest')

    @property
    def training_mask(self):
        mask = self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))
        print(f"DEBUG: BaseAdapter.training_mask shape: {mask.shape}")
        return mask

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        # Return the first adjacency matrix
        if self.adjs is None:
            return None
        adj = self.adjs if self.adjs.ndim == 2 else self.adjs[0]
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        return adj

    def splitter(self, dataset, val_len=1., in_sample=False, window=0):
        nontest_idxs, test_idxs = disjoint_months(dataset, months=self.test_months, synch_mode='horizon')
        if in_sample:
            train_idxs = np.arange(len(dataset))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = disjoint_months(dataset, months=val_months, synch_mode='horizon')
        else:
            # take equal number of samples before each month of testing
            val_len = (int(val_len * len(nontest_idxs)) if val_len < 1 else val_len) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][np.flatnonzero(delta_idxs > delta_idxs.min())]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs]
            val_idxs = np.concatenate(month_val_idxs) % len(dataset)
            # remove overlapping indices from training set
            ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs, val_idxs, synch_mode='horizon', as_mask=True)
            train_idxs = nontest_idxs[~ovl_idxs]
        return [train_idxs, val_idxs, test_idxs]

    def load_data(self):
        raise NotImplementedError


class Beijing12Adapter(BaseAdapter):
    def __init__(self, **kwargs):
        super().__init__('beijing12', freq='h', **kwargs)

    def load_data(self):
        base_path = datasets_path['beijing12']
        path = os.path.join(base_path, 'beijing12.h5')

        try:
            df = pd.DataFrame(pd.read_hdf(path, 'pm25'))
        except ValueError as e:
            if "unrecognized index type" in str(e):
                print(f"WARNING: Standard load failed due to index version mismatch. Attempting raw load...")
                import tables
                with tables.open_file(path, mode='r') as h5file:
                    try:
                        node = h5file.get_node('/pm25')
                    except tables.NoSuchNodeError:
                        node = h5file.get_node('/data')
                    try:
                        values = node.block0_values.read()
                        try:
                            columns = node.axis0.read().astype(str)
                        except:
                            columns = None
                        start_date = pd.Timestamp('2013-03-01 00:00:00')
                        index = pd.date_range(start=start_date, periods=values.shape[0], freq='h')
                        if columns is not None:
                            df = pd.DataFrame(values, index=index, columns=columns)
                        else:
                            df = pd.DataFrame(values, index=index)
                        print("Successfully reconstructed DataFrame with inferred index.")
                    except Exception as fallback_e:
                        print(f"Fallback load failed: {fallback_e}")
                        raise e
            else:
                raise e

        adjs = np.load(os.path.join(base_path, 'adjacency.npy'))
        positions = np.load(os.path.join(base_path, 'position.npy'))
        adj_label = np.load(os.path.join(base_path, 'adjacency_label.npy'))

        return df, adjs, positions, adj_label


class VesselAdapter(BaseAdapter):
    def __init__(self, **kwargs):
        # Override test_months for Vessel because it only covers a short period (January only?)
        # Vessel data length is 1800 steps * 10s = 5 hours?
        # If dataset spans < 1 year, disjoint_months based on [3,6,9,12] might return empty.
        super().__init__('vessel', freq='10S', **kwargs)
        # Vessel dataset is very short (1800 samples = 5 hours).
        # Standard monthly split won't work.
        # Use simple chronological split or single month if data falls in one month.
        self.test_months = [1]  # Assuming data is in Jan, or just force 1 month if index is synthetic Jan

    def splitter(self, dataset, val_len=0.1, test_len=0.2, in_sample=False, window=0):
        # Override splitter for Vessel to use simple time split instead of month-based
        # because the dataset duration is too short (hours/days).
        n_samples = len(dataset)
        test_len_samples = int(n_samples * test_len)
        val_len_samples = int(n_samples * val_len)
        train_len_samples = n_samples - test_len_samples - val_len_samples

        train_idxs = np.arange(train_len_samples)
        val_idxs = np.arange(train_len_samples, train_len_samples + val_len_samples)
        test_idxs = np.arange(train_len_samples + val_len_samples, n_samples)

        return [train_idxs, val_idxs, test_idxs]

    def load_data(self):
        base_path = datasets_path['vessel']
        path = os.path.join(base_path, 'vessel_ais.h5')

        df = pd.read_hdf(path, 'data')

        # Reconstruct index for Vessel
        start_date = pd.Timestamp('2024-01-01')
        new_index = start_date + pd.to_timedelta(df.index * 10, unit='s')
        df.index = new_index

        adjs = np.load(os.path.join(base_path, 'adjacency.npy'))
        positions = np.load(os.path.join(base_path, 'position.npy'))
        adj_label = np.load(os.path.join(base_path, 'adjacency_label.npy'))

        return df, adjs, positions, adj_label
