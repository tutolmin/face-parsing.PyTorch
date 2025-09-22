#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    #     0: 'background', 1: 'skin', 2: 'brows', 3: 'eyes', 4: 'nose', 5: 'mouth', 6: 'u_lip', 7: 'l_lip',
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85],
                   [0, 255, 0], 
                   [0, 255, 170],
                   [255, 0, 85], 
                   [0, 0, 255], [85, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [255, 0, 170],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def save_individual_masks(parsing_anno, image_path, base_filename):
    """
    Сохраняет индивидуальные маски для каждого класса в формате CelebAMask-HQ
    
    Args:
        parsing_anno: маска с семантической сегментацией (numpy array)
        image_path: полный путь к исходному изображению
        base_filename: базовое имя файла без расширения
    """
    # Соответствие индексов классов и их имен
    class_mapping = {
        0: 'background',
        1: 'skin', 
        2: 'brows',
        3: 'eyes',
        4: 'eye_g',  # очки
        5: 'nose',
        6: 'mouth', 
        7: 'u_lip',
        8: 'l_lip',
        9: 'tongue',
    }
    
    # Создаем папку для масок, если она не существует
    mask_dir = '/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno-tongue'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    # Сохраняем маски для каждого класса
    for class_idx, class_name in class_mapping.items():
        if class_name == 'background':  # пропускаем фон
            continue
            
        # Проверяем, есть ли пиксели данного класса в маске
        class_pixels = np.sum(parsing_anno == class_idx)
        if class_pixels == 0:  # если класс не содержит пикселей, пропускаем
            continue        
            
        # Создаем бинарную маску для текущего класса
        binary_mask = (parsing_anno == class_idx).astype(np.uint8) * 255
        
        # Создаем имя файла в формате: filename_mask.png
        mask_filename = f"{base_filename}___{class_name}.png"
        mask_path = osp.join(mask_dir, mask_filename)
        
        # Сохраняем маску как PNG
        mask_img = Image.fromarray(binary_mask)
        mask_img.save(mask_path)

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 9
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
#            image = img.resize((512, 512), Image.BILINEAR)
            image = img.resize((1024, 1024), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


            # Добавляем вызов функции для сохранения индивидуальных масок
            base_filename = osp.splitext(image_path)[0]  # получаем имя файла без расширения
            save_individual_masks(parsing, image_path, base_filename)


if __name__ == "__main__":
#    evaluate(dspth='/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img', cp='29999_iter.pth')
#    evaluate(dspth='/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img', cp='4999_iter.pth')
    evaluate(dspth='/home/andrei/data/job4/images', cp='49999_iter.pth')


