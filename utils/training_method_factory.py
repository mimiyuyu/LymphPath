def create_evaluation(cfg):
    if cfg.Train.val_function == 'lymphpath':
        from training_methods.lymphpath import evaluation
    elif cfg.Train.val_function == 'classification_single':
        from training_methods.classification_single import evaluation
    else:
        raise NotImplementedError
    return evaluation


def create_training_loop(cfg):
    if cfg.Train.train_function == 'lymphpath_train':
        from training_methods.lymphpath import train_loop
    else:
        raise NotImplementedError
    return train_loop


def create_validation(cfg):
    if cfg.Train.val_function == 'lymphpath':
        from training_methods.lymphpath import validation
    else:
        raise NotImplementedError
    return validation
