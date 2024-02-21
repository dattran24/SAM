from ..scheduler import *
import torch
import os
def get_scheduler(optimizer, cfg):
    if cfg['trainer']['sch'] == "constant_lr":
        return torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=cfg['trainer']['factor'],
        total_iters=cfg['trainer']['total_iters']
    )
    elif cfg['trainer']['sch'] == "diminish1":
        return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: 1/(epoch+1)
    )
    elif cfg['trainer']['sch'] == "diminish2":
        return Diminish2(
            optimizer,
            learning_rate=cfg['model']['lr']
        )
    elif cfg['trainer']['sch'] == "diminish3":
        return Diminish3(
            optimizer,
            learning_rate=cfg['model']['lr']
        )
    elif cfg['trainer']['sch'] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['trainer']['epochs'], eta_min=0, last_epoch=-1, verbose='deprecated'
        )
    else:
        raise ValueError("Invalid scheduler!!!")