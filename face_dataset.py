#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

from transform import *



class FaceMask(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth

        # Список атрибутов и их объединённые ID
        self.atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.merged_ids = {
            'skin': 1,
            'l_brow': 2, 'r_brow': 2,
            'l_eye': 3, 'r_eye': 3,
            'nose': 4,
            'mouth': 5,
            'u_lip': 6,
            'l_lip': 7,
            'eye_g': 8  # новый класс
        }

        self.imgs = os.listdir(os.path.join(self.rootpth, 'CelebA-HQ-img'))
        total_imgs = len(self.imgs)
        self.partial_start_idx = total_imgs - 1445  # последние 1445 — только eye_g размечен

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(osp.join(self.rootpth, 'CelebA-HQ-img', impth)).convert('RGB')
        img = img.resize((512, 512), Image.BILINEAR)

        # Инициализация маски как "игнорировать всё"
        label = np.full((512, 512), self.ignore_lb, dtype=np.int64)

        is_partial = idx >= self.partial_start_idx

        if is_partial:
            # Только eye_g размечен
            path = osp.join(self.rootpth, 'CelebAMask-HQ-mask_eye_g', f"{idx}.png")
            if osp.exists(path):
                mask_eye_g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                label[mask_eye_g == 8] = 8  # только пиксели eye_g активны
        else:
            # Полная разметка: собираем все классы
            for att in self.atts:
                path = osp.join(self.rootpth, 'CelebAMask-HQ-mask-anno', str(idx // 2000),
                                f"{str(idx).zfill(5)}_{att}.png")
                if not osp.exists(path):
                    continue
                try:
                    mask = np.array(Image.open(path).convert('P'))
                    pixels = (mask == 225) | (mask == 255)
                    if np.any(pixels):
                        class_id = self.merged_ids.get(att)
                        if class_id is not None:
                            label[pixels] = class_id
                except:
                    pass

        # Преобразуем в PIL для аугментаций
        label = Image.fromarray(label.astype(np.uint8), mode='P')

        # Аугментации
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    face_data = '/home/andrei/data/celebmaskhq/CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = '/home/andrei/data/celebmaskhq/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = '/home/andrei/data/CelebAMask-HQ/mask'
    counter = 0
    total = 0
    for i in range(15):
        # files = os.listdir(osp.join(face_sep_mask, str(i)))

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i*2000, (i+1)*2000):

            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))

                    mask[sep_mask == 225] = l
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)

    print(counter, total)














