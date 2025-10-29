from torch.utils.data import DataLoader
import pandas as pd

def create_dataloader(dataset, cfg, result_dir):
    if cfg.Data.dataset_name in ['ThreeStreamBagDataset']:
        return create_bag_dataloader(dataset, cfg, result_dir)
    else:
        raise NotImplementedError

def create_bag_dataloader(dataset_name, cfg, result_dir):
    df = pd.read_csv(cfg.Data.dataset_test)
    if cfg.Data.dataset_name == 'ThreeStreamBagDataset':
        from datasets.ThreeStreamBagDataset import ThreeChannelBagDataset
        dataset = ThreeChannelBagDataset(df, **cfg.Data)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                    num_workers=cfg.Train.num_worker)
    else:
        raise NotImplementedError
    return dataloader