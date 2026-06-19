import os
import random
import shutil
import math

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler


def stratified_sampling(cfg, result_dir):
    cv_root = os.path.join(result_dir, 'cross_validation_splits')
    os.makedirs(cv_root, exist_ok=True)
    df = pd.read_csv(cfg.Data.dataset['train_set'].csv_path)

    labels = set(df['label'].values)
    data = []
    for label in labels:
        class_data = list(df.values[df['label'].values == label])
        random.shuffle(class_data)
        data.append(class_data)

    fold_num = cfg.General.fold_num
    for i in range(fold_num):
        fold_data = []
        for class_data in data:
            fold_size = int(math.ceil(len(class_data) / fold_num + 0.5))
            fold_data += class_data[fold_size * i: fold_size * i + fold_size]
        pd.DataFrame(data=np.array(fold_data), columns=df.columns).to_csv(
            os.path.join(cv_root, f'split_{i}.csv'), index=False
        )


def _read_split_csv(cfg, dataset_name):
    if dataset_name in ['train_set', 'val_set']:
        return pd.read_csv(cfg.Data.dataset[dataset_name].csv_path)

    if 'dataset_test' in cfg.Data:
        test_csv = cfg.Data.dataset_test
        return pd.read_csv(test_csv if isinstance(test_csv, str) else test_csv.csv_path)

    test_set = cfg.Data.dataset[dataset_name]
    if isinstance(test_set, str):
        return pd.read_csv(test_set)
    return pd.read_csv(test_set.csv_path)


def create_dataloader(index, dataset_name, cfg, result_dir):
    if cfg.Data.dataset_name == 'BagDataset':
        return _create_bag_dataloader(index, dataset_name, cfg, result_dir)
    if cfg.Data.dataset_name != 'ThreeStreamBagDataset':
        raise NotImplementedError

    if dataset_name in ['train_set', 'val_set']:
        if cfg.General.exp_type == 'repeat':
            df = pd.read_csv(cfg.Data.dataset[dataset_name].csv_path)
        else:
            if isinstance(cfg.Data.split_dir, str) and not os.path.exists(
                os.path.join(result_dir, 'cross_validation_splits')
            ):
                shutil.copytree(cfg.Data.split_dir, os.path.join(result_dir, 'cross_validation_splits'))

            if not os.path.exists(os.path.join(result_dir, 'cross_validation_splits')):
                stratified_sampling(cfg, result_dir)
            if dataset_name == 'train_set':
                df_parts = []
                for i in range(cfg.General.fold_num):
                    if i != index:
                        df_parts.append(pd.read_csv(
                            os.path.join(result_dir, 'cross_validation_splits', f'split_{i}.csv')
                        ))
                df = pd.concat(df_parts, axis=0)
            else:
                df = pd.read_csv(os.path.join(result_dir, 'cross_validation_splits', f'split_{index}.csv'))
    else:
        df = _read_split_csv(cfg, dataset_name)

    from datasets.ThreeStreamBagDataset import ThreeChannelBagDataset
    dataset = ThreeChannelBagDataset(df, **cfg.Data)

    if dataset_name == 'train_set' and cfg.Train.balance:
        weights = dataset.get_balance_weight()
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            sampler=ExhaustiveWeightedRandomSampler(weights, len(weights)),
            num_workers=cfg.Train.num_worker,
            pin_memory=True,
        )
    elif dataset_name == 'train_set':
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=True,
            num_workers=cfg.Train.num_worker,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=cfg.Train.num_worker,
        )

    return dataloader


def _create_bag_dataloader(index, dataset_name, cfg, result_dir):
    if dataset_name in ['train_set', 'val_set']:
        df = pd.read_csv(cfg.Data.dataset[dataset_name].csv_path)
    else:
        df = _read_split_csv(cfg, dataset_name)

    from datasets.BagDataset import BagDataset
    data_kwargs = dict(cfg.Data)
    data_dir = data_kwargs.pop('data_dir')
    for key in ('dataset', 'dataset_name', 'n_classes'):
        data_kwargs.pop(key, None)
    dataset = BagDataset(df, data_dir, **data_kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.Train.num_worker,
    )
    return dataloader
