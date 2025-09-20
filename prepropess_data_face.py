#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
import numpy as np
from PIL import Image

face_sep_mask = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask_face'
counter = 0
total = 0

# Новые классы после объединения
merged_atts = ['face']

# Маппинг старых классов на новые
class_mapping = {
    'skin': 'face',
    'l_brow': 'face',
    'r_brow': 'face',
    'l_eye': 'face',
    'r_eye': 'face',
    'nose': 'face',
    'mouth': 'face',
    'u_lip': 'face',
    'l_lip': 'face'
}

for i in range(15):
    # Исходные атрибуты
    original_atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                     'nose', 'mouth', 'u_lip', 'l_lip']

    for j in range(i * 2000, (i + 1) * 2000):
        mask = np.zeros((512, 512))

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

        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        print(j)

print(counter, total)
print(f"Итоговые классы: {merged_atts}")
print(f"Количество классов: {len(merged_atts)} (было {len(original_atts)})")

