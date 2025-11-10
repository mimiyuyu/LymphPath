import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils.utils import read_yaml
from utils.dataloader_factory import create_dataloader
from utils.model_factory import load_model
from utils.training_method_factory import create_evaluation
import csv

def calculate_metrics(result_dir, cfg, decimals=4):
    """Calculate AUC and AUPRC metrics"""
    pred_file = os.path.join(result_dir, f'preds_0.csv')
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return None

    df = pd.read_csv(pred_file)
    label = df['label'].values
    prob = df['prob_1'].values

    # Calculate AUC
    auc_score = np.around(roc_auc_score(label, prob), decimals=decimals)

    # Calculate AUPRC
    precision1, recall1, _ = precision_recall_curve(label, prob)
    auprc_score = np.around(auc(recall1, precision1), decimals=decimals)

    return auc_score, auprc_score


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/task0_sample.yaml')
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
    test_csv_paths = cfg.Data.dataset_test
    test_dir_name = test_csv_paths.split('/')[-1].split('.')[0]
    ckpt_dir = './weights/'

    result_dir = os.path.join(cfg.General.result_dir, test_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    dataloader = create_dataloader(args.dataset_name, cfg, result_dir)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'LymphPath.pt')))
    evaluation(model, dataloader, result_dir, cfg)

    # Compute metrics
    print(f"\nCalculating metrics for {test_dir_name} ...")
    auc_score, auprc_score = calculate_metrics(result_dir, cfg, args.decimals)

    # Create result DataFrame (one row: dataset, AUC, AUPRC)
    df = pd.DataFrame([{
        'dataset': test_dir_name,
        'AUC': auc_score,
        'AUPRC': auprc_score
    }])

    print(f"\nResults for {test_dir_name}:")
    print(df)

    # Save results
    metrics_file = os.path.join(result_dir, 'metrics.csv')
    df.to_csv(metrics_file, index=False, encoding='gbk')
    print(f"Metrics saved to: {metrics_file}")
