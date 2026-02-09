import copy
import datetime
import os
import pathlib
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.rwr import get_position
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool


def has_graph_support(model_cls):
    return False


def get_model_classes(model_str):
    if model_str == 'adactsi_fullwindow':
        model = models.AdaCTSiFullWindow
    elif model_str == 'adactsi':
        model = models.AdaCTSi
    else:
        raise ValueError(f"Model {model_str} not available.")
    filler = fillers.Filler
    return model, filler


def get_dataset(args, dataset_name):
    if dataset_name == 'pems_point':
        n_nodes = args.n_nodes if args.n_nodes > 0 else None
        dataset = datasets.MissingValuesPems(args.subdataset_name, p_fault=0., p_noise=args.p_missing, n_nodes=n_nodes)
    elif dataset_name == 'beijing12':
        from lib.datasets.dataset_adapter import Beijing12Adapter
        dataset = Beijing12Adapter(impute_nans=True)
    elif dataset_name == 'vessel':
        from lib.datasets.dataset_adapter import VesselAdapter
        limit_nodes = getattr(args, 'limit_nodes', None)
        if limit_nodes is None:
            limit_nodes = 20
        dataset = VesselAdapter(impute_nans=True, limit_nodes=limit_nodes)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--model-name", type=str, default='adactsi')
    parser.add_argument("--dataset-name", type=str, default='pems_point')
    parser.add_argument("--subdataset-name", type=str, default='PEMS11')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=400)

    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5)
    parser.add_argument('--grad-clip-algorithm', type=str, default='value')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)
    parser.add_argument('--steps-per-day', type=int, default=288)
    parser.add_argument('--n-nodes', type=int, default=64,
                        help='Number of nodes to sample for PEMS datasets. Set to -1 for all nodes.')
    parser.add_argument('--limit-nodes', type=int, default=None, help='Limit nodes for debugging (e.g. Vessel)')
    parser.add_argument('--p-missing', type=float, default=0.25, help='Missing rate for PEMS/Covid datasets')
    # Inference complexity control (100 means full complexity, 25 means 25% of queries)
    parser.add_argument('--inference-complexity', type=int, nargs='+', default=[100], choices=[25, 50, 75, 100],
                        help='Inference complexity percentage (25, 50, 75, 100). Provide one or more values.')
    parser.add_argument('--sfa-limit-nodes', type=int, nargs='+', default=None,
                        help='List of limit_nodes for SFA (Sensor Failure Adaptability) experiments. '
                             'If provided, will override limit-nodes and run multiple experiments.')
    parser.add_argument('--inference-strategy', type=str, default='default',
                        choices=['default', 'cw', 'arw', 'random', 'drift', 'group'],
                        help='Inference strategy to use. "default" uses standard forward. '
                             '"cw" uses Codebook/Correlation Weighted mechanism.')

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    if known_args.config is not None:
        with open(known_args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        parser.set_defaults(**config_args)

    args = parser.parse_args()
    print(args)
    return args


class PeriodicValidationMetrics(Callback):
    def __init__(self, every_n_epochs=20):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return
        metrics = trainer.callback_metrics
        keys = ['val_loss', 'val_mae', 'val_mape', 'val_mre', 'val_mse']
        parts = []
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if hasattr(value, 'item'):
                    value = value.item()
                parts.append(f"{key}={value:.4f}")
        if parts:
            print(f"Epoch {epoch}: " + ", ".join(parts))


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args, args.dataset_name)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    print(dataset.numpy().shape, dataset.eval_mask.shape, dataset.training_mask.shape)
    t_mask = dataset.training_mask
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=t_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride,
                                adjs=dataset.adjs,
                                positions=dataset.positions,
                                adj_label=dataset.adj_label)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)
    positions, anchors = get_position(adj)

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in // 3, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')
    periodic_callback = PeriodicValidationMetrics(every_n_epochs=20)

    logger = TensorBoardLogger(logdir, name="model")
    # logger = None
    print('grad_clip_algorithm', args.grad_clip_algorithm)
    accelerator = 'cpu'
    devices = 1
    gpu_arg = str(args.gpu).strip().lower()
    if gpu_arg in {'-1', 'cpu', 'none'}:
        accelerator = 'cpu'
        devices = 1
    elif gpu_arg == 'auto':
        if torch.cuda.is_available():
            accelerator = 'gpu'
            devices = 1
        else:
            accelerator = 'cpu'
            devices = 1
    else:
        gpu_ids = [int(x) for x in gpu_arg.split(',') if x.strip()]
        if gpu_ids:
            accelerator = 'gpu'
            devices = gpu_ids
    version_parts = (pl.__version__ or "0.0").split(".")
    major = int(version_parts[0]) if version_parts[0].isdigit() else 0
    minor = int(version_parts[1]) if len(version_parts) > 1 and version_parts[1].isdigit() else 0
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        logger=logger,
        default_root_dir=logdir,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=args.grad_clip_val,
        gradient_clip_algorithm=args.grad_clip_algorithm,
        callbacks=[early_stop_callback, checkpoint_callback, periodic_callback],
        num_sanity_val_steps=0,
    )
    if major > 1 or (major == 1 and minor >= 5):
        trainer_kwargs["enable_progress_bar"] = False
        trainer_kwargs["enable_model_summary"] = False
    else:
        trainer_kwargs["progress_bar_refresh_rate"] = 0
        trainer_kwargs["weights_summary"] = None
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(filler, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test(datamodule=dm, verbose=False)
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    # Run evaluation for different complexity levels
    complexities = sorted(list(set(args.inference_complexity)))
    final_summary = {}

    for complexity in complexities:
        print(f"\nEvaluating with Inference Complexity: {complexity}% and Strategy: {args.inference_strategy}")

        # Set complexity
        if hasattr(filler.model, 'inference_complexity'):
            filler.model.inference_complexity = complexity
        else:
            filler.model.inference_complexity = complexity

        # Set inference strategy
        filler.model.inference_strategy = args.inference_strategy
        filler.model.limit_nodes = args.limit_nodes

        with torch.no_grad():
            y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)
        y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

        # Test imputations in whole series
        eval_mask = dataset.eval_mask[dm.test_slice]
        df_true = dataset.df.iloc[dm.test_slice]
        metrics = {
            'mae': numpy_metrics.masked_mae,
            'mse': numpy_metrics.masked_mse,
            'mre': numpy_metrics.masked_mre,
            'mape': numpy_metrics.masked_mape
        }
        # Aggregate predictions in dataframes
        index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
        aggr_methods = ensure_list(args.aggregate_by)
        third_of_columns = dataset.df.shape[1] // 3
        df_hats = prediction_dataframe(y_hat, index, dataset.df.columns[:third_of_columns], aggregate_by=aggr_methods)
        df_hats = dict(zip(aggr_methods, df_hats))
        output = defaultdict(list)
        for aggr_by, df_hat in df_hats.items():
            # Compute error
            print(f'- AGGREGATE BY {aggr_by.upper()}')
            for metric_name, metric_fn in metrics.items():
                error = metric_fn(df_hat.values, df_true.values[:, :third_of_columns], eval_mask).item()
                print(f' {metric_name}: {error:.4f}')
                output[str(metric_name)].append(error)

        final_summary[complexity] = output

    print("\n\n=======================================================")
    print("FINAL SUMMARY (All Complexities)")
    print("=======================================================")
    for complexity in complexities:
        print(f"\nComplexity: {complexity}%")
        # Assuming 'mean' aggregation is the primary one or first one
        # The output dict has list of values for each metric (corresponding to aggr_methods)
        # We print all of them
        res = final_summary[complexity]
        for metric, values in res.items():
            # values is a list corresponding to aggr_methods
            # Let's just print them
            if len(values) == 1:
                print(f"  {metric.upper()}: {values[0]:.4f}")
            else:
                print(f"  {metric.upper()}: {values}")

    return final_summary


if __name__ == '__main__':
    # Expect datasets to live under the project root: ./datasets
    current_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(current_dir, 'datasets')
    if not os.path.isdir(datasets_dir) or not os.listdir(datasets_dir):
        print(f"Warning: datasets folder missing or empty at {datasets_dir}")

    args = parse_args()

    # Check for SFA batch experiment
    # We want to run this if sfa_limit_nodes is present AND (we explicitly asked for it via CLI OR strategy is drift)
    # If strategy is NOT drift, but sfa_limit_nodes is present (e.g. from YAML), do we run it?
    # Maybe only if user explicitly requested it or if strategy implies it.

    if args.sfa_limit_nodes and (args.inference_strategy == 'drift' or 'sfa_limit_nodes' in sys.argv):
        print(f"\n[SFA] Detected SFA experiment with limit_nodes: {args.sfa_limit_nodes}")
        sfa_results = {}

        # Sort descending to keep consistency (or ascending? User listed [10, 8, 6], decreasing)
        # We will follow the order provided in config/args
        nodes_list = args.sfa_limit_nodes

        for n_nodes in nodes_list:
            print(f"\n\n[SFA] >>>>> Running Experiment with limit_nodes={n_nodes} <<<<<")
            current_args = copy.deepcopy(args)
            current_args.limit_nodes = n_nodes
            # Clear sfa_limit_nodes to prevent recursion if we passed args again (though run_experiment takes args object)
            current_args.sfa_limit_nodes = None

            # Run experiment
            # Note: run_experiment returns a dict of results keyed by complexity
            exp_result = run_experiment(current_args)
            sfa_results[n_nodes] = exp_result

        print("\n\n#######################################################")
        print("SFA EXPERIMENT FINAL SUMMARY")
        print("#######################################################")

        for n_nodes in nodes_list:
            print(f"\n[Limit Nodes: {n_nodes}]")
            res_by_complexity = sfa_results[n_nodes]

            # Iterate through complexities (usually just [100] or whatever was requested)
            for complexity, metrics in res_by_complexity.items():
                print(f"  Complexity {complexity}%:")
                for metric_name, values in metrics.items():
                    if len(values) == 1:
                        print(f"    {metric_name.upper()}: {values[0]:.4f}")
                    else:
                        print(f"    {metric_name.upper()}: {values}")

        outs = sfa_results
    else:
        outs = run_experiment(args)

    print(outs)
