import torch


def load_model(cfg):
    if cfg.Model.model_type != 'stable':
        raise NotImplementedError
    model = load_stable_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    return model


def load_stable_model(cfg):
    if cfg.Model.model_name == 'LymphPath':
        from models.stable_models.LymphPath import LymphPath
        model = LymphPath(n_classes=cfg.Data.n_classes, **cfg.Model)
        print('success to init LymphPath')
    elif cfg.Model.model_name == 'CLAM_SB':
        from models.stable_models.LymphPath import CLAM_SB
        model = CLAM_SB(n_classes=cfg.Data.n_classes, **cfg.Model)
        print('success to init CLAM_SB')
    else:
        raise NotImplementedError
    return model
