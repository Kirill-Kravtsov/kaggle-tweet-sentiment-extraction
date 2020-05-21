import os
import yaml
import pydoc
import argparse
import logging
from copy import deepcopy
from collections import OrderedDict
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import (CriterionCallback, MetricAggregationCallback,
                                   SchedulerCallback, OptimizerCallback,
                                   CheckpointCallback)
from tokenizers import ByteLevelBPETokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
from datasets import TweetDataset
from losses import QACrossEntropyLoss
from collators import DynamicPaddingCollator
from callbacks import JaccardCallback, SWACallback, SheduledDropheadCallback
from utils import seed_torch, processify


TRAINING_DEFAULTS = {
    'main_metric': 'jaccard',
    'num_epochs': 3,
    'verbose': False,
    'minimize_metric': False,
    'valid_loader': "valid_swa"
}


DEFAULT_CALLBACKS = {
    'swa': {
        '__class__': "callbacks.SWACallback",
        'swa_start': 0,
        'swa_freq': 2
    },
    'criterion': {
        '__class__': "catalyst.dl.callbacks.CriterionCallback",
        'input_key': ["start_positions", "end_positions"],
        'output_key': ["start_logits", "end_logits"]
    },
    'jaccard': {
        '__class__': "callbacks.JaccardCallback"
    },
    'scheduler': {
        '__class__': "catalyst.dl.callbacks.SchedulerCallback",
        'mode': "batch"
    },
    'checkpoint': {
        '__class__': "callbacks.CustomCheckpointCallback",
        'save_n_best': 10
    }
}


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--val-fold', type=int, default=None)
    parser.add_argument('--ignore-fold', type=int, default=None)
    parser.add_argument('--logdir', type=str, default="../logs")

    args = parser.parse_args()

    assert (args.cv is False) or (args.val_fold is None)

    if args.model_name is None:
        args.model_name = os.path.splitext(os.path.basename(args.config))[0]
    #args.model_name += '_cv%i_fold%i'%(args.num_folds, args.val_fold)
    if args.debug:
        args.model_name += "_debug"
    if args.ignore_fold is not None:
        args.model_name = os.path.join(args.model_name,
                                       f"ignore_{args.ignore_fold}")
    return args


def create_class_obj(dct, get_by_key=None, default_cls=None,
                     overwite_config = False, **kwargs):
    # get class dct if needed to get by key
    dct = deepcopy(dct)
    if get_by_key is not None:
        if get_by_key in dct:
            dct = dct[get_by_key]
        else:
            dct = {}

    # determine class
    if '__class__' in dct:
        if type(dct['__class__']) == str:
            cls = pydoc.locate(dct['__class__'])
            if cls is None:
                logging.warning(f"Class {dct['__class__']} is not found")
        else:  # assumes class definition object was provided
            cls = dct['__class__']
        del dct['__class__']
    else:
        cls = default_cls
    if cls is None:
        return None

    # maybe it should be created by some class method
    if '__by_method__' in dct:
        cls = getattr(cls, dct['__by_method__'])
        del dct['__by_method__']

    # get params and create object
    if dct.get('__overwrite_defaults__', False):
        del dct['__overwrite_defaults__']
    else:
        for k, v in kwargs.items():
            if (k not in dct) or (overwite_config):
                dct[k] = v

    return cls(**dct)


def create_callbacks(callback_dct):
    callbacks = OrderedDict()
    for name, args in callback_dct.items():
        callbacks[name] = create_class_obj(args)
    return callbacks


def clear_checkpoints(logdir, best_epoch):
    files = glob.glob(os.path.join(logdir, "*", "checkpoints", "*"))
    for path in files:
        if ("metrics" not in path) and ("best" not in path) \
            and ("train."+str(best_epoch) not in path
        ):
            os.remove(path)


@processify  # because of some gpu memory leak
def run_fold(config, args, train_folds, val_fold):
    print(f"Val fold: {val_fold}, train_folds: {train_folds}")
    model_path = config['model']['pretrained_model_name_or_path']

    tokenizer = create_class_obj(
        config,
        get_by_key='tokenizer',
        default_cls=ByteLevelBPETokenizer,
        vocab_file=os.path.join(model_path, "vocab.json"), 
        merges_file=os.path.join(model_path, "merges.txt"),
        lowercase=True,
        add_prefix_space=True
    )
    train_data = create_class_obj(
        config,
        get_by_key='dataset',
        default_cls=TweetDataset,
        folds=train_folds,
        tokenizer=tokenizer,
        max_num_samples = 100 if args.debug else None
    )
    valid_data = create_class_obj(
        config,
        get_by_key='dataset',
        default_cls=TweetDataset,
        folds=val_fold,
        tokenizer=tokenizer,
        max_num_samples = 100 if args.debug else None
    )

    collator = create_class_obj(
        config,
        get_by_key='collator',
        default_cls=DynamicPaddingCollator
    )

    train_loader = create_class_obj(
        config,
        get_by_key='dataloader',
        default_cls=DataLoader,
        dataset=train_data,
        batch_size=16,
        num_workers=8,
        shuffle=True
    )
    valid_loader = create_class_obj(
        config,
        get_by_key='dataloader',
        default_cls=DataLoader,
        dataset=valid_data,
        batch_size=16,
        num_workers=8,
        shuffle=False
    )

    logging.info(
        f'Train #batches {len(train_loader)}, val #batches {len(valid_loader)}'
    )

    model = create_class_obj(config['model']).cuda()
    runner = create_class_obj(
        config,
        get_by_key='runner',
        default_cls=SupervisedRunner,
        input_key=['ids', 'mask', 'token_type_ids'],
        input_target_key=['start_positions', 'end_positions'],
        output_key=['start_logits', 'end_logits']
    )

    criterion = create_class_obj(
        config,
        get_by_key='criterion',
        default_cls=QACrossEntropyLoss
    )

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = create_class_obj(
        config,
        get_by_key='optimizer',
        default_cls=AdamW,
        params=optimizer_parameters,
        lr=3e-5
    )

    train_params = config.get('train_params', {})
    for k, v in TRAINING_DEFAULTS.items():
        if k not in train_params:
            train_params[k] = v
    logdir = os.path.join(args.logdir, args.model_name, f"fold_{val_fold}")

    scheduler = create_class_obj(
        config,
        get_by_key='scheduler',
        default_cls=get_linear_schedule_with_warmup,
        optimizer=optimizer,
        num_training_steps=len(train_loader) * train_params['num_epochs'],
        num_warmup_steps=len(train_loader) * train_params['num_epochs']//10
    )

    callbacks = create_callbacks(config.get('callbacks', DEFAULT_CALLBACKS))

    runner.train(
        model=model,
        loaders={'train': train_loader, 'valid': valid_loader, 'valid_swa': valid_loader},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        logdir=logdir,
        **train_params
    )

    with open(os.path.join(logdir, "checkpoints", "_metrics.json"), "r") as f:
        metrics = json.load(f)
    metrics = {k:v[2]['jaccard'] for k, v in metrics.items() if "epoch" in k}
    return metrics


def main():
    seed_torch()
    args = parse_args()
    config = get_config(args.config)
    folds = list(set(pd.read_csv(config['dataset']['df_path'])['fold']))
    if args.ignore_fold is not None:
        folds.remove(args.ignore_fold)
    if args.val_fold:
        avg_metrics = run_fold(config, args, args.val_fold)
    elif args.cv:
        avg_metrics = {}
        best_metrics = []
        best_epochs = {}
        for val_fold in folds:
            train_folds = [i for i in folds if i!=val_fold]
            metrics = run_fold(config, args, train_folds, val_fold)
            metrics = {int(k.split("_")[1])+1:v for k, v in metrics.items()}
            if len(avg_metrics) == 0:
                avg_metrics = {k:[v] for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    avg_metrics[k].append(v)
            best_metrics.append(np.max(list(metrics.values())))
            best_epochs[val_fold] = max(metrics, key=metrics.get)

        avg_metrics = {k:np.average(v) for k, v in avg_metrics.items()}
        avg_metrics['best_avg_epoch'] = max(avg_metrics, key=avg_metrics.get)
        avg_metrics['best'] = np.average(best_metrics)
        avg_metrics['best_epochs'] = best_epochs

        clear_checkpoints(os.path.join(args.logdir, args.model_name),
                          avg_metrics['best_avg_epoch'])
        path = os.path.join(args.logdir, args.model_name, "avg_metrics.json")
        with open(path, 'w') as f:
            json.dump(avg_metrics, f)
    print(avg_metrics)


if __name__ == "__main__":
    main()
