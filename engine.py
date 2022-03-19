# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.nn.functional as F
import utils
from torch import nn



def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, mcp=False):
    model.train(set_training_mode)    #i just turn off for training trainable layers only
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.10f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for img0, img1, labels in metric_logger.log_every(data_loader, print_freq, header):
        img0, img1 , labels = img0.cuda(), img1.cuda() , labels.cuda()
        '''
        x1, x2 = model(img0, img1)
        loss = criterion(x1, x2, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        '''
        with torch.cuda.amp.autocast(enabled=not fp32):
            output = model(img0, img1, labels)
            loss = criterion(output, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)   # changed this for trainable layers
                    #parameters=trainable_layers.parameters(), create_graph=is_second_order) 

        torch.cuda.synchronize()

        

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
