#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
import numpy as np
from PIL import Image

face_sep_mask = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask_eye_g_100_full'
counter = 0
total = 0

# Новые классы после объединения
merged_atts = ['skin', 'brows', 'eyes', 'nose', 'mouth', 'u_lip', 'l_lip', 'eye_g']

# Маппинг старых классов на новые
class_mapping = {
    'skin': 'skin',
    'l_brow': 'brows',
    'r_brow': 'brows',
    'l_eye': 'eyes',
    'r_eye': 'eyes',
    'nose': 'nose',
    'mouth': 'mouth',
    'u_lip': 'u_lip',
    'l_lip': 'l_lip',
    'eye_g': 'eye_g'
}

# Находим индексы nose и eye_g (с учетом enumerate от 1)
nose_index = merged_atts.index('nose') + 1  # 4
eye_g_index = merged_atts.index('eye_g') + 1  # 8

for i in range(15):
    # Исходные атрибуты
    original_atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                     'nose', 'mouth', 'u_lip', 'l_lip', 'eye_g']

    for j in range(i * 2000, (i + 1) * 2000):
        mask = np.zeros((512, 512))
        flag = False

        for att in original_atts:
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))

                # Определяем новый индекс класса
                new_class = class_mapping[att]
                new_idx = merged_atts.index(new_class) + 1  # +1 т.к. фон = 0

                mask[sep_mask == 225] = new_idx

                if att == 'eye_g':
                    flag = True
        if flag:
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)

print(counter, total)
print(f"Итоговые классы: {merged_atts}")
print(f"Количество классов: {len(merged_atts)} (было {len(original_atts)})")

