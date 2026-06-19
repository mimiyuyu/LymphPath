import os

import pandas as pd
import torch
from tqdm import tqdm


def evaluation(index, model, loader, result_dir, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    probs = []
    labels = []

    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for batch in loader:
                x = batch['x'].to(device, dtype=torch.float32)
                y = batch['y'].to(device, dtype=torch.long)

                result = model(x)
                logits = result[cfg.Model.logits_field] if cfg.Model.logits_field in result else result['bag_logits']
                y_prob = torch.softmax(logits, dim=-1)

                probs.append(y_prob)
                labels.append(y)
                bar.update(1)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    id_list = loader.dataset.get_id_list()

    df_dict = {'id': id_list, 'label': labels}
    for i in range(cfg.Data.n_classes):
        df_dict[f'prob_{i}'] = probs[:, i]
    pd.DataFrame(df_dict).to_csv(os.path.join(result_dir, f'preds_{index}.csv'), index=False)
