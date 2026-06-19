import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, df, data_dir, label_field='label', **kwargs):
        super(BagDataset, self).__init__()
        self.data_dir = data_dir
        self.df = df
        self.label_field = label_field

    def __len__(self):
        return len(self.df.values)

    def __getitem__(self, idx):
        label = self.df[self.label_field].values[idx]
        slide_id = self.df['slide_id'].values[idx]

        if 'dataset' in self.df.columns:
            dataset_name = self.df['dataset'].values[idx]
        else:
            dataset_name = 'default'

        if 'feature_path' in self.df.columns:
            full_path = self.df['feature_path'].values[idx]
        else:
            if os.path.exists(os.path.join(self.data_dir, 'patch_feature')):
                full_path = os.path.join(self.data_dir, 'patch_feature', slide_id + '.pt')
            else:
                full_path = os.path.join(self.data_dir, slide_id + '.pt')
        features = torch.load(full_path, map_location=torch.device('cpu'))

        return {
            'x': features,
            'y': torch.tensor([label]),
            'dataset_name': dataset_name,
        }

    def get_id_list(self):
        return self.df['slide_id'].values
