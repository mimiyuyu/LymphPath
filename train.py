import argparse
import os
import shutil

import torch
from tensorboardX import SummaryWriter

from utils.dataloader_factory import create_dataloader
from utils.model_factory import load_model
from utils.optimizer_factory import create_optimizer
from utils.training_method_factory import create_training_loop, create_validation
from utils.utils import EarlyStopping, read_yaml, save_task_state, seed_torch


def training_task(index, result_dir, cfg):
    model = load_model(cfg)
    train_loader = create_dataloader(index, 'train_set', cfg, result_dir)
    val_loader = create_dataloader(index, 'val_set', cfg, result_dir)

    print('#' * 30 + '\n'
          f'load dataset successfully!\n'
          f'train set size: {len(train_loader.dataset)}\n'
          f'val set size: {len(val_loader.dataset)}\n'
          + '#' * 30)

    writer_dir = os.path.join(result_dir, 'log', str(index))
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    optimizer = create_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.Train.CosineAnnealingLR.T_max,
        eta_min=cfg.Train.CosineAnnealingLR.eta_min,
        last_epoch=-1,
    )

    early_stopping = EarlyStopping(
        patience=cfg.Train.Early_stopping.patient,
        stop_epoch=cfg.Train.Early_stopping.stop_epoch,
        type='min' if cfg.Train.Early_stopping.type in ['loss'] else 'max',
    )

    train_loop = create_training_loop(cfg)
    validation = create_validation(cfg)

    print('#' * 10 + ' training task start! ' + '#' * 10)

    for epoch in range(cfg.Train.max_epochs):
        save_task_state(result_dir, early_stopping, epoch, model, index)
        lr = scheduler.get_last_lr()[0]
        print('learning rate:{:.8f}'.format(lr))
        writer.add_scalar('train/lr', lr, epoch)

        train_loop(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            writer=writer,
            cfg=cfg,
            result_dir=result_dir,
        )

        with torch.no_grad():
            stop = validation(
                index=index,
                epoch=epoch,
                model=model,
                loader=val_loader,
                result_dir=result_dir,
                early_stopping=early_stopping,
                writer=writer,
                cfg=cfg,
            )

        if stop:
            break
        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del train_loader, val_loader
        train_loader = create_dataloader(index, 'train_set', cfg, result_dir)
        val_loader = create_dataloader(index, 'val_set', cfg, result_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--set_seed', action='store_true')
    args = parser.parse_args()

    cfg = read_yaml(args.config_path)
    config_name = os.path.basename(args.config_path)
    result_dir = os.path.join(cfg.General.result_dir, f'{config_name}_seed{cfg.General.seed}')
    os.makedirs(result_dir, exist_ok=True)

    config_copy = os.path.join(result_dir, config_name)
    if not os.path.exists(config_copy):
        shutil.copy(args.config_path, result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(args.begin, args.end):
        if args.set_seed:
            seed_torch(device, cfg.General.seed + i)
        training_task(i, result_dir, cfg)
