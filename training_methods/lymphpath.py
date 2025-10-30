import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, auc

def evaluation(model, loader, result_dir, cfg):
    n_classes = cfg.Data.n_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for idx, batch in enumerate(loader):
                feature1, feature2, feature3, y = batch['features1'].to(device, dtype=torch.float32), \
                batch['features2'].to(device, dtype=torch.float32), \
                batch['features3'].to(device, dtype=torch.float32), \
                batch['label'].to(device, dtype=torch.long)

                result = model(feature1, feature2, feature3, label=y, instance_eval=True) # (1, n_classes)

                logits1 = result['logits1']
                logits2 = result['logits2']
                logits3 = result['logits3']
                bag_logits_merge = result['merge_logits']

                loss1 = loss_fn(logits1, y)
                loss2 = loss_fn(logits2, y)
                loss3 = loss_fn(logits3, y)
                merge_loss = loss_fn(bag_logits_merge, y)
                loss = cfg.Model.loss1 * (loss1 + loss2 + loss3) / 3 + cfg.Model.mergeloss * merge_loss

                y_prob = result['y_prob']
                probs.append(y_prob)
                labels.append(y)

                val_loss += loss.item()
                bar.update(1)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    id_list = loader.dataset.get_id_list()

    df_dict = {'id': id_list, 'label': labels}
    for i in range(n_classes):
        df_dict[f'prob_{i}'] = probs[:, i]
    pd.DataFrame(df_dict).to_csv(os.path.join(result_dir, f'preds_0.csv'), index=False)
    

