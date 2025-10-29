def create_evaluation(cfg):
    if cfg.Train.val_function == 'lymphpath':
        from training_methods.lymphpath import evaluation
    else:
        raise NotImplementedError

    return evaluation