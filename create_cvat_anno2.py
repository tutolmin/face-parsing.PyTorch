import os
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import glob

def create_cvat_annotation(mask_dir, output_file):
    """
    Создает файл аннотации CVAT 1.1 из бинарных масок
    
    Args:
        mask_dir (str): Путь к каталогу с масками
        output_file (str): Путь к выходному XML файлу
    """
    
    # Классы для аннотирования
    classes = ['skin', 'brows', 'eyes', 'eye_g', 'nose', 'mouth', 'u_lip', 'l_lip']
    
    # Создаем корневой элемент
    annotations = Element('annotations')
    
    # Добавляем метаинформацию
    version = SubElement(annotations, 'version')
    version.text = '1.1'
    
    # Создаем элемент meta
    meta = SubElement(annotations, 'meta')
    
    # Создаем элемент task внутри meta
    task = SubElement(meta, 'task')
    
    # Основная информация о задаче
    SubElement(task, 'name').text = 'CelebAMask-HQ-tongue-annotation'
    SubElement(task, 'size').text = '0'  # Будет обновлено позже
    SubElement(task, 'mode').text = 'annotation'
    SubElement(task, 'overlap').text = '0'
    SubElement(task, 'bugtracker').text = ''
    SubElement(task, 'created').text = '2024-01-01 00:00:00'  # Замените на актуальную дату
    SubElement(task, 'updated').text = '2024-01-01 00:00:00'  # Замените на актуальную дату
    
    # Размер изображения
    labels = SubElement(task, 'labels')
    
    # Добавляем метки для каждого класса
    for class_name in classes:
        label = SubElement(labels, 'label')
        SubElement(label, 'name').text = class_name
        # Можно добавить дополнительные атрибуты для каждой метки при необходимости
    
    # Собираем все файлы масок
    mask_files = glob.glob(os.path.join(mask_dir, '*_*.png'))
    
    # Группируем маски по имени изображения
    image_masks = {}
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        # Извлекаем имя изображения (например, '012_0' из '012_0_eye_g.png')
        parts = filename.split('_')
        if len(parts) >= 3:
            image_name = f"{parts[0]}_{parts[1]}"
            # Определяем класс из имени файла
            class_part = '_'.join(parts[2:]).replace('.png', '')
            
            if class_part in classes:
                if image_name not in image_masks:
                    image_masks[image_name] = []
                image_masks[image_name].append((class_part, mask_file))
    
    # Счетчик для id изображений
    image_id = 0
    
    # Обрабатываем каждое изображение
    for image_name, masks in image_masks.items():
        image_filename = f"{image_name}.png"
        
        # Создаем элемент image
        image_elem = SubElement(annotations, 'image')
        image_elem.set('id', str(image_id))
        image_elem.set('name', image_filename)
        image_elem.set('width', '1024')
        image_elem.set('height', '1024')
        
        # Обрабатываем каждую маску для этого изображения
        for class_name, mask_path in masks:
            # Загружаем маску
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Предупреждение: не удалось загрузить маску {mask_path}")
                continue
            
            # Находим контуры в маске
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Обрабатываем каждый контур
            for contour in contours:
                # Упрощаем контур (уменьшаем количество точек)
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Если контур слишком мал, пропускаем его
                if len(approx) < 3:
                    continue
                
                # Создаем полигон
                polygon = SubElement(image_elem, 'polygon')
                polygon.set('label', class_name)
                polygon.set('source', 'semi-auto')
                polygon.set('occluded', '0')
                polygon.set('z_order', '0')
                
                # Формируем строку с точками
                points_str = ""
                for point in approx:
                    x, y = point[0]
                    points_str += f"{x:.2f},{y:.2f};"
                
                # Убираем последнюю точку с запятой
                points_str = points_str[:-1]
                polygon.set('points', points_str)
        
        image_id += 1
    
    # Обновляем количество изображений в метаданных
    size_elem = task.find('size')
    if size_elem is not None:
        size_elem.text = str(len(image_masks))
    
    # Сохраняем XML файл
    xml_str = prettify(annotations)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"Аннотация сохранена в {output_file}")
    print(f"Обработано изображений: {len(image_masks)}")

def prettify(elem):
    """Возвращает красиво отформатированную XML строку"""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

if __name__ == "__main__":
    # Укажите путь к каталогу с масками
    mask_directory = "CelebAMask-HQ-mask-anno-tongue"
    
    # Укажите имя выходного файла
    output_filename = "cvat_annotations.xml"
    
    # Создаем аннотацию
    create_cvat_annotation(mask_directory, output_filename)
