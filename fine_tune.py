#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from evaluate import evaluate
from optimizer import Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse


respth = './res_finetune'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parse.add_argument("--pretrained-weights", type=str, required=True,
                      help="Path to pretrained model weights")
    parse.add_argument("--new-classes", type=int, default=1,
                      help="Number of new classes to add")
    parse.add_argument("--freeze-backbone", action="store_true",
                      help="Freeze backbone layers during fine-tuning")
    parse.add_argument("--lr-multiplier", type=float, default=0.1,
                      help="Learning rate multiplier for fine-tuning")
    return parse.parse_args()

def modify_model_for_new_classes(net, n_old_classes, n_new_classes):
    """
    Modify the model to accommodate new classes by expanding the final layer
    """
    # Get the original convolutional layer
    original_conv = net.conv_out
    
    # Create new convolutional layer with expanded output channels
    new_out_channels = n_old_classes + n_new_classes
    
    new_conv = nn.Conv2d(
        original_conv.in_channels,
        new_out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=(original_conv.bias is not None)
    )
    
    # Initialize weights for new classes
    nn.init.normal_(new_conv.weight, mean=0, std=0.01)
    if new_conv.bias is not None:
        nn.init.constant_(new_conv.bias, 0)
    
    # Copy weights from old classes
    with torch.no_grad():
        new_conv.weight[:n_old_classes] = original_conv.weight.clone()
        if new_conv.bias is not None:
            new_conv.bias[:n_old_classes] = original_conv.bias.clone()
    
    # Replace the convolutional layer
    net.conv_out = new_conv
    
    return net

def freeze_backbone_layers(net):
    """
    Freeze backbone layers for fine-tuning only the classifier
    """
    # Freeze all layers except the final convolutional layer
    for name, param in net.named_parameters():
        if 'conv_out' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return net

def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    setup_logger(respth)

    # Configuration
    n_old_classes = 8  # Original number of classes
    n_new_classes = args.new_classes
    n_classes = n_old_classes + n_new_classes
    n_img_per_gpu = 16  # Smaller batch size for fine-tuning
    n_workers = 4
    cropsize = [512, 512]  # Smaller crop size for fine-tuning
    data_root = '/home/andrei/data/CelebAMask-HQ/'

    # Load dataset with new classes
    ds = FaceMask(data_root, cropsize=cropsize, mode='train', 
                 include_new_classes=True, n_new_classes=n_new_classes)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    # Load and modify model
    net = BiSeNet(n_classes=n_old_classes)
    
    # Load pretrained weights
    state_dict = torch.load(args.pretrained_weights, map_location='cpu')
    net.load_state_dict(state_dict)
    
    # Modify model for new classes
    net = modify_model_for_new_classes(net, n_old_classes, n_new_classes)
    
    net.cuda()
    net.train()
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        net = freeze_backbone_layers(net)
    
    net = nn.parallel.DistributedDataParallel(net,
            device_ids=[args.local_rank],
            output_device=args.local_rank)

    # Loss function
    ignore_idx = -100
    score_thres = 0.7
    n_min_p = n_img_per_gpu * (cropsize[0]//8) * (cropsize[1]//8) // 16
    n_min_2 = n_img_per_gpu * (cropsize[0]//16) * (cropsize[1]//16) // 16  
    n_min_3 = n_img_per_gpu * (cropsize[0]//32) * (cropsize[1]//32) // 16

    LossP = OhemCELoss(thresh=score_thres, n_min=n_min_p, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min_2, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min_3, ignore_lb=ignore_idx)

    # Optimizer with reduced learning rate
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 5e-3 * args.lr_multiplier  # Reduced learning rate
    max_iter = 20000  # Fewer iterations for fine-tuning
    power = 0.9
    warmup_steps = 500
    warmup_start_lr = 1e-5 * args.lr_multiplier
    
    # Only optimize parameters that require gradients
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    
    optim = Optimizer(
        model=trainable_params,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    # Training loop
    scaler = GradScaler()
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    
    for it in range(max_iter):
        try:
            im, lb = next(diter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
            
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            out, out16, out32 = net(im)

            # Interpolate to original label size
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            out16 = F.interpolate(out16, size=(H, W), mode='bilinear', align_corners=False)
            out32 = F.interpolate(out32, size=(H, W), mode='bilinear', align_corners=False)

            lossp = LossP(out, lb)
            loss2 = Loss2(out16, lb)
            loss3 = Loss3(out32, lb)

            loss = lossp + loss2 + loss3
            
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_avg.append(loss.item())

        # Print training log
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it=it+1,
                    max_it=max_iter,
                    lr=lr,
                    loss=loss_avg,
                    time=t_intv,
                    eta=eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
            
        if dist.get_rank() == 0:
            if (it+1) % 2000 == 0:
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state, './res_finetune/cp/finetune_{}_iter.pth'.format(it))
                evaluate(dspth='/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img_eye_g', 
                        cp='finetune_{}_iter.pth'.format(it))

    # Save final model
    save_pth = osp.join(respth, 'model_finetuned.pth')
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('Fine-tuning done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    try:
        train()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

