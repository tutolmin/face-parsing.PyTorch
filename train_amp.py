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


respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--local-rank", "--local_rank", type=int, default = 0)
#    parse.add_argument(
#            '--local_rank',
#            dest = 'local_rank',
#            type = int,
#            default = 0,
#            )
    return parse.parse_args()

# First, define class weights based on importance
def get_class_weights():
    # CelebAMask-HQ has 19 classes: 
    # 0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 
    # 6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth, 
    # 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    
#    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

#    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
#            'nose', 'mouth', 'u_lip', 'l_lip']

#    merged_atts = ['skin', 'brows', 'eyes', 'nose', 'mouth', 'u_lip', 'l_lip']

    # Weight values (higher = more important)
    weights = torch.ones(9)  # default weight is 1
    
    # Very important classes (eyes, mouth, lips)
    weights[3] = 3.0   # eyes
    weights[5] = 3.0  # mouth
    weights[6] = 3.0  # u_lip
    weights[7] = 3.0  # l_lip

    # Still important
    weights[2] = 2.0   # brows
    weights[4] = 2.0  # nose
    
    # Moderately important
    weights[1] = 1.0   # skin

    # Default
#    weights[17] = 1.0  # hair
#    weights[0] = 1.0  # background
    
    # Less important (set to <1 to reduce their impact)
#    weights[7] = 0.0   # l_ear
#    weights[8] = 0.0   # r_ear
#    weights[14] = 0.0  # neck

    # Exclude
    weights[8] = 2.0   # eye_g (eyeglasses)
#    weights[9] = 0.0   # ear_r (earrings)
#    weights[15] = 0.0  # neck_l (necklace)
#    weights[16] = 0.0  # cloth
#    weights[18] = 0.0  # hat
    
    return weights.cuda()


def finetune():
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

    # === Изменения по сравнению с train() ===
    n_classes = 9  # Теперь 9 классов
    ignore_idx = -100  # Уже используется
    cp_path = './res/model_final_diss.pth'  # Путь к сохранённой модели
    # =======================================

    n_img_per_gpu = 32
    n_workers = 8
    cropsize = [448, 448]
    data_root = '/home/andrei/data/CelebAMask-HQ/'  # Должен включать новые данные

    ds = FaceMask(data_root, cropsize=cropsize, mode='finetune')  # или 'train', но с новыми данными
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    # model
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    # Загружаем предобученные веса
    state_dict = torch.load(cp_path, map_location=torch.device('cpu'))
    # Совместимость: удалить префикс 'module.' если нужно
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        new_state_dict[nk] = v
    net.load_state_dict(new_state_dict, strict=False)  # strict=False — т.к. добавился новый класс

    net.train()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids=[args.local_rank],
            output_device=args.local_rank)

    score_thres = 0.7
    n_min_p = n_img_per_gpu * (cropsize[0]//8) * (cropsize[1]//8) // 16
    n_min_2 = n_img_per_gpu * (cropsize[0]//16) * (cropsize[1]//16) // 16
    n_min_3 = n_img_per_gpu * (cropsize[0]//32) * (cropsize[1]//32) // 16

    # Веса классов с учётом нового класса
    class_weight = get_class_weights()
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min_p, ignore_lb=ignore_idx, weight=class_weight)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min_2, ignore_lb=ignore_idx, weight=class_weight)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min_3, ignore_lb=ignore_idx, weight=class_weight)

    # Оптимизатор — можно уменьшить lr для fine-tuning
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-4  # Меньше, чем при обучении с нуля
    max_iter = 20000  # Меньше итераций
    power = 0.9
    warmup_steps = 200
    warmup_start_lr = 1e-6

    optim = Optimizer(
        model=net.module,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    # === Обучение ===
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

        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / (it + 1)))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                'finetune',
                'it: {it}/{max_it}',
                'lr: {lr:.6f}',
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
            if (it+1) % 5000 == 0 or (it+1) == max_iter:
                state = net.module.state_dict()
                torch.save(state, f'./res/cp/ft_{it}_iter.pth')
#                evaluate(dspth='/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img', cp=f'ft_{it}_iter.pth')
                evaluate(dspth='/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img_eye_g', cp=f'ft_{it}_iter.pth')

    # Сохранение финальной модели
    save_pth = osp.join(respth, 'model_final_finetuned.pth')
    state = net.module.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info(f'Fine-tuning done, model saved to: {save_pth}')


def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:29500',
#                init_method = 'tcp://127.0.0.1:33241',
                world_size = torch.cuda.device_count(),
#                rank=0
                rank=args.local_rank
                )
    setup_logger(respth)

    # dataset
#    n_classes = 19
    n_classes = 9
    n_img_per_gpu = 32
    n_workers = 8
    cropsize = [448, 448]
#    cropsize = [512, 512]
#    cropsize = [704, 704]
#    cropsize = [768, 768]
#    cropsize = [1024, 1024]
    data_root = '/home/andrei/data/CelebAMask-HQ/'

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    # model
    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank],
            output_device = args.local_rank
            )
    score_thres = 0.7

    # Предполагая, что out: /8, out16: /16, out32: /32
    n_min_p = n_img_per_gpu * (cropsize[0]//8) * (cropsize[1]//8) // 16
    n_min_2 = n_img_per_gpu * (cropsize[0]//16) * (cropsize[1]//16) // 16  
    n_min_3 = n_img_per_gpu * (cropsize[0]//32) * (cropsize[1]//32) // 16

#    LossP = OhemCELoss(thresh=score_thres, n_min=n_min_p, ignore_lb=ignore_idx)
#    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min_2, ignore_lb=ignore_idx)
#    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min_3, ignore_lb=ignore_idx)

    class_weight = get_class_weights()

    LossP = OhemCELoss(thresh=score_thres, n_min=n_min_p, ignore_lb=ignore_idx, weight=class_weight)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min_2, ignore_lb=ignore_idx, weight=class_weight)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min_3, ignore_lb=ignore_idx, weight=class_weight)

#    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
##    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
##    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
##    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#
#    # Modify the loss initialization
#    class_weight = get_class_weights()
#    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=class_weight)
#    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=class_weight)
#    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=class_weight)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
#    lr_start = 1e-2
    lr_start = 5e-3
    max_iter = 100000
    power = 0.9
#    power = 0.6
    warmup_steps = 1000
#    warmup_steps = 500
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net.module,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
#    scaler = GradScaler('cuda')
    scaler = GradScaler()
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
# Redundant
#            if not im.size()[0] == n_img_per_gpu:
#                raise StopIteration
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
#        out, out16, out32 = net(im)
#        lossp = LossP(out, lb)
#        loss2 = Loss2(out16, lb)
#        loss3 = Loss3(out32, lb)
#        loss = lossp + loss2 + loss3
#        loss.backward()

        with autocast(device_type='cuda', dtype=torch.float16):
            out, out16, out32 = net(im)

            # Интерполировать до размера оригинальных меток
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            out16 = F.interpolate(out16, size=(H, W), mode='bilinear', align_corners=False)
            out32 = F.interpolate(out32, size=(H, W), mode='bilinear', align_corners=False)

            lossp = LossP(out, lb)
            loss2 = Loss2(out16, lb)
            loss3 = Loss3(out32, lb)

#        with autocast(device_type='cuda', dtype=torch.float16):
#            out, out16, out32 = net(im)
#            lossp = LossP(out, lb)
#            loss2 = Loss2(out16, lb)
#            loss3 = Loss3(out32, lb)

            loss = lossp + loss2 + loss3
        scaler.scale(loss).backward()
        scaler.step(optim)   # ← теперь не упадёт
        scaler.update()

        loss_avg.append(loss.item())

        #  print training log message
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
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        if dist.get_rank() == 0:
            if (it+1) % 5000 == 0:
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#                if dist.get_rank() == 0:
                torch.save(state, './res/cp/{}_iter.pth'.format(it))
                evaluate(dspth='/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img', cp='{}_iter.pth'.format(it))

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    # net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
#    train()
    try:
#        train()
        finetune()  # ← вместо train()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
