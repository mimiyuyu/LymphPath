import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.dataloader_factory import create_dataloader
from utils.model_factory import load_model
from utils.training_method_factory import create_evaluation
from utils.utils import read_yaml


def calculate_metrics(result_dir, decimals=4):
    pred_file = os.path.join(result_dir, 'preds_0.csv')
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return None

    df = pd.read_csv(pred_file)
    label = df['label'].values
    prob = df['prob_1'].values

    auc_score = np.around(roc_auc_score(label, prob), decimals=decimals)
    auprc_score = np.around(average_precision_score(label, prob), decimals=decimals)
    return auc_score, auprc_score


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--checkpoint', type=str, default='./checkpoints/LymphPath.pt')
parser.add_argument('--dataset_name', type=str, default='test_set')
parser.add_argument('--decimals', type=int, default=4)
args = parser.parse_args()

if __name__ == '__main__':
    import sys
    if sys.platform != 'linux':
        args.config_path = args.config_path.replace('\\', '/')

    cfg = read_yaml(args.config_path)
    model = load_model(cfg)
    evaluation = create_evaluation(cfg)

    if 'dataset_test' in cfg.Data:
        test_csv_path = cfg.Data.dataset_test
    elif 'test_set' in cfg.Data.dataset:
        test_set = cfg.Data.dataset.test_set
        test_csv_path = test_set if isinstance(test_set, str) else test_set.csv_path
    else:
        raise ValueError("Config must define Data.dataset_test or Data.dataset.test_set")

    test_dir_name = os.path.splitext(os.path.basename(test_csv_path))[0]
    result_dir = os.path.join(cfg.General.result_dir, test_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    dataloader = create_dataloader(0, args.dataset_name, cfg, result_dir)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    evaluation(0, model, dataloader, result_dir, cfg)

    print(f"\nCalculating metrics for {test_dir_name} ...")
    auc_score, auprc_score = calculate_metrics(result_dir, args.decimals)

    df = pd.DataFrame([{
        'dataset': test_dir_name,
        'AUC': auc_score,
        'AUPRC': auprc_score,
    }])

    print(f"\nResults for {test_dir_name}:")
    print(df)

    metrics_file = os.path.join(result_dir, 'metrics.csv')
    df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to: {metrics_file}")
