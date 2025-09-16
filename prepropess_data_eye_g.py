#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
from transform import *
from PIL import Image

face_sep_mask = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno_orig'
mask_path = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask_eye_g'
counter = 0
total = 0
for i in range(15):

#    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    atts = ['eye_g']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))
        flag = False

        for l, att in enumerate(atts, 8):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                flag = True
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l

        if flag:
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)

print(counter, total)
