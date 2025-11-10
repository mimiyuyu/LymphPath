import os

import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ThreeChannelBagDataset(Dataset):
    def __init__(self, df, data_dir1, data_dir2, data_dir3, label_field='label', **kwargs):
        super(ThreeChannelBagDataset, self).__init__()

        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.df = df
        self.label_field = label_field

    def __len__(self):
        return len(self.df.values)

    def __getitem__(self, idx):
        label = self.df[self.label_field].values[idx]
        slide_id = self.df['slide_id'].values[idx]

        # load from pt files
        if os.path.exists(os.path.join(self.data_dir1, 'patch_feature')):
            full_path1 = os.path.join(self.data_dir1, 'patch_feature', slide_id + '.pt')
        else:
            full_path1 = os.path.join(self.data_dir1, slide_id + '.pt')
        features1 = torch.load(full_path1, map_location=torch.device('cpu'))

        if os.path.exists(os.path.join(self.data_dir2, 'patch_feature')):
            full_path2 = os.path.join(self.data_dir2, 'patch_feature', slide_id + '.pt')
        else:
            full_path2 = os.path.join(self.data_dir2, slide_id + '.pt')
        features2 = torch.load(full_path2, map_location=torch.device('cpu'))

        if os.path.exists(os.path.join(self.data_dir3, 'patch_feature')):
            full_path3 = os.path.join(self.data_dir3, 'patch_feature', slide_id + '.pt')
        else:
            full_path3 = os.path.join(self.data_dir3, slide_id + '.pt')
        features3 = torch.load(full_path3, map_location=torch.device('cpu'))

        res = {
            'features1': features1,
            'features2': features2,
            'features3': features3,
            'label': torch.tensor([label]),
        }
        return res

    def get_id_list(self):
        return self.df['slide_id'].values