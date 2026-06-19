import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from utils.ablation import compute_bag_loss, compute_pred_prob, get_active_streams


def train_loop(epoch, model, loader, optimizer, writer, cfg, result_dir, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_streams = get_active_streams(cfg)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_weights = []
    model.to(device)

    with tqdm(total=len(loader), desc='train epoch: {}'.format(epoch)) as bar:
        for idx, batch in enumerate(loader):
            feature1 = batch['features1'].to(device, dtype=torch.float32)
            feature2 = batch['features2'].to(device, dtype=torch.float32)
            feature3 = batch['features3'].to(device, dtype=torch.float32)
            y = batch['label'].to(device, dtype=torch.long)

            result = model(
                feature1, feature2, feature3,
                label=y, instance_eval=True, active_streams=active_streams,
            )
            inst_loss = result['inst_loss']
            bag_logits1 = result['logits1']
            bag_logits2 = result['logits2']
            bag_logits3 = result['logits3']
            bag_logits_merge = result['merge_logits']
            branch_weights = result.get('branch_weights')

            if branch_weights is not None:
                epoch_weights.append(branch_weights.detach().cpu().numpy())

            bag_loss = compute_bag_loss(
                loss_fn, bag_logits1, bag_logits2, bag_logits3, bag_logits_merge, y, cfg, active_streams,
            )
            loss = 0.8 * bag_loss + 0.2 * inst_loss

            bar.set_postfix({'loss': '{:.5f}'.format(loss)})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer:
                writer.add_scalar('train/loss', loss, epoch * len(loader) + idx)
            bar.update(1)

    if epoch_weights:
        mean_weights = np.array(epoch_weights).mean(axis=0)
        print(f"Epoch {epoch} Weights: {mean_weights}")

        save_path = os.path.join(result_dir, "branch_weights.csv")
        df = pd.DataFrame([{
            'epoch': epoch,
            'w1': mean_weights[0],
            'w2': mean_weights[1],
            'w3': mean_weights[2],
        }])
        if not os.path.exists(save_path):
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, mode='a', header=False, index=False)


def _compute_metrics(y_true, y_prob, n_classes):
    acc = accuracy_score(y_true, np.argmax(y_prob, axis=-1).astype(int))
    f1 = f1_score(y_true, np.argmax(y_prob, axis=-1).astype(int), average='weighted')
    if n_classes == 2:
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        auprc_score = average_precision_score(y_true, y_prob[:, 1])
    else:
        auc_score = 0
        auprc_score = 0
        for i in range(n_classes):
            auc_score += roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            auprc_score += average_precision_score((y_true == i).astype(int), y_prob[:, i])
        auc_score /= n_classes
        auprc_score /= n_classes
    return acc, f1, auc_score, auprc_score


def validation(index, epoch, model, loader, result_dir, early_stopping, writer, cfg):
    n_classes = cfg.Data.n_classes
    early_stopping_type = cfg.Train.Early_stopping.type
    active_streams = get_active_streams(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    stream_weights_list = []
    dataset_groups = {}

    with torch.no_grad():
        with tqdm(total=len(loader), desc='validate epoch:{}'.format(epoch)) as bar:
            for batch in loader:
                feature1 = batch['features1'].to(device, dtype=torch.float32)
                feature2 = batch['features2'].to(device, dtype=torch.float32)
                feature3 = batch['features3'].to(device, dtype=torch.float32)
                y = batch['label'].to(device, dtype=torch.long)
                dataset_name = batch['dataset_name']

                result = model(
                    feature1, feature2, feature3,
                    label=y, instance_eval=True, active_streams=active_streams,
                )
                logits1 = result['logits1']
                logits2 = result['logits2']
                logits3 = result['logits3']
                bag_logits_merge = result['merge_logits']
                stream_weights = result.get('stream_weights')
                if stream_weights is not None:
                    stream_weights_list.append(stream_weights.detach().cpu())

                loss = compute_bag_loss(
                    loss_fn, logits1, logits2, logits3, bag_logits_merge, y, cfg, active_streams,
                )
                y_prob = compute_pred_prob(cfg, logits1, logits2, logits3, bag_logits_merge, active_streams)

                probs.append(y_prob)
                labels.append(y)

                if dataset_name not in dataset_groups:
                    dataset_groups[dataset_name] = {'labels': [], 'probs': []}
                dataset_groups[dataset_name]['labels'].append(y)
                dataset_groups[dataset_name]['probs'].extend(y_prob.cpu().numpy())

                val_loss += loss.item()
                bar.update(1)

    val_loss /= len(loader)

    if len(dataset_groups) == len(loader):
        y_true = torch.cat(labels, dim=0).cpu().numpy()
        y_prob = torch.cat(probs, dim=0).cpu().numpy()
        acc, f1, aucc, auprc = _compute_metrics(y_true, y_prob, n_classes)
    else:
        aucc = 0
        auprc = 0
        acc = 0
        f1 = 0
        for values in dataset_groups.values():
            y_true = torch.cat(values['labels'], dim=0).cpu().numpy()
            y_prob = np.array(values['probs'])
            group_acc, group_f1, group_auc, group_auprc = _compute_metrics(y_true, y_prob, n_classes)
            acc = group_acc
            f1 = group_f1
            aucc += group_auc
            auprc += group_auprc
        aucc /= len(dataset_groups)
        auprc /= len(dataset_groups)

    if stream_weights_list:
        avg_stream_weights = torch.cat(stream_weights_list, dim=0).mean(dim=0).numpy()
        print('Avg stream weights: stream1 {:.4f}, stream2 {:.4f}, stream3 {:.4f}'.format(
            avg_stream_weights[0], avg_stream_weights[1], avg_stream_weights[2]
        ))

    print('\nVal Set, val_loss: {:.4f}, f1: {:.4f}, acc {:.4f}, auc: {:.4f}, auprc: {:.4f}'.format(
        val_loss, f1, acc, aucc, auprc
    ))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', acc, epoch)

    if early_stopping:
        os.makedirs(os.path.join(result_dir, 'best_checkpoints'), exist_ok=True)
        ckpt_name = os.path.join(result_dir, 'best_checkpoints', "s_{}_checkpoint.pt".format(index))
        if early_stopping_type == 'loss':
            early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'acc':
            early_stopping(epoch, acc, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'auprc':
            early_stopping(epoch, auprc, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'auc':
            early_stopping(epoch, aucc, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'f1':
            early_stopping(epoch, f1, model, ckpt_name=ckpt_name)
        else:
            raise NotImplementedError
        if early_stopping.early_stop:
            print('Early stopping')
            return True

    return False


def evaluation(index, model, loader, result_dir, cfg):
    n_classes = cfg.Data.n_classes
    active_streams = get_active_streams(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    probs = []
    labels = []
    stream_weights_list = []

    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for batch in loader:
                feature1 = batch['features1'].to(device, dtype=torch.float32)
                feature2 = batch['features2'].to(device, dtype=torch.float32)
                feature3 = batch['features3'].to(device, dtype=torch.float32)
                y = batch['label'].to(device, dtype=torch.long)

                result = model(
                    feature1, feature2, feature3,
                    label=y, instance_eval=True, active_streams=active_streams,
                )
                logits1 = result['logits1']
                logits2 = result['logits2']
                logits3 = result['logits3']
                bag_logits_merge = result['merge_logits']
                stream_weights = result.get('stream_weights')

                y_prob = compute_pred_prob(cfg, logits1, logits2, logits3, bag_logits_merge, active_streams)

                probs.append(y_prob)
                labels.append(y)
                if stream_weights is not None:
                    stream_weights_list.append(stream_weights.detach().cpu())
                bar.update(1)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    id_list = loader.dataset.get_id_list()

    df_dict = {'id': id_list, 'label': labels}
    for i in range(n_classes):
        df_dict[f'prob_{i}'] = probs[:, i]
    if stream_weights_list:
        stream_weights = torch.cat(stream_weights_list, dim=0).numpy()
        for i in range(stream_weights.shape[1]):
            df_dict[f'stream_weight_{i + 1}'] = stream_weights[:, i]
    pd.DataFrame(df_dict).to_csv(os.path.join(result_dir, f'preds_{index}.csv'), index=False)
