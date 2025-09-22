import os
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import glob
from collections import defaultdict

def create_cvat_annotation(mask_dir, output_file):
    """
    Создает файл аннотации CVAT 1.1 из бинарных масок
    
    Args:
        mask_dir (str): Путь к каталогу с масками
        output_file (str): Путь к выходному XML файлу
    """
    
    # Классы для аннотирования
    classes = ['skin', 'brows', 'eyes', 'eye_g', 'nose', 'mouth', 'u_lip', 'l_lip', 'tongue']
    
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
    SubElement(task, 'created').text = '2024-01-01 00:00:00'
    SubElement(task, 'updated').text = '2024-01-01 00:00:00'
    
    # Размер изображения
    labels = SubElement(task, 'labels')
    
    # Добавляем метки для каждого класса
    for class_name in classes:
        label = SubElement(labels, 'label')
        SubElement(label, 'name').text = class_name
    
    # Собираем все файлы масок
    mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
    
    # Словарь для группировки масок по имени изображения
    image_masks = defaultdict(list)
    
    # Множество всех уникальных изображений
    all_image_names = set()
    
    print(f"Найдено файлов масок: {len(mask_files)}")
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        
        # Разделяем имя файла по тройному подчеркиванию
        if '___' in filename:
            parts = filename.split('___')
            if len(parts) == 2:
                filename_part = parts[0]  # Часть с именем файла
                class_part = parts[1].replace('.png', '')  # Часть с классом (убираем .png)
                
                # Имя изображения - это filename_part
                image_name = filename_part
                all_image_names.add(image_name)
                
                if class_part in classes:
                    image_masks[image_name].append((class_part, mask_file))
                else:
                    print(f"Предупреждение: неизвестный класс '{class_part}' в файле {filename}")
            else:
                print(f"Предупреждение: неверный формат имени файла {filename}")
        else:
            print(f"Предупреждение: файл {filename} не содержит тройного подчеркивания")
    
    print(f"Найдено уникальных изображений: {len(all_image_names)}")
    print(f"Изображений с аннотациями: {len(image_masks)}")
    
    # Счетчик для id изображений
    image_id = 0
    
    # Обрабатываем каждое изображение (даже если для него есть только одна маска)
    for image_name in sorted(all_image_names):
        image_filename = f"{image_name}.png"
        
        # Создаем элемент image
        image_elem = SubElement(annotations, 'image')
        image_elem.set('id', str(image_id))
        image_elem.set('name', image_filename)
        image_elem.set('width', '1024')
        image_elem.set('height', '1024')
        
        # Обрабатываем маски для этого изображения (если они есть)
        masks = image_masks.get(image_name, [])
        
        for class_name, mask_path in masks:
            # Загружаем маску
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Предупреждение: не удалось загрузить маску {mask_path}")
                continue
            
            # Проверяем размер маски
            if mask.shape != (1024, 1024):
                print(f"Предупреждение: маска {mask_path} имеет размер {mask.shape}, ожидается (1024, 1024)")
                # Пропускаем или обрабатываем в зависимости от требований
                continue
            
            # Находим контуры в маске
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Обрабатываем каждый контур
            contour_count = 0
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
                contour_count += 1
            
            if contour_count > 0:
                print(f"  Добавлен класс {class_name} с {contour_count} контурами")
        
        image_id += 1
        print(f"Обработано изображение: {image_filename} (масок: {len(masks)})")
    
    # Обновляем количество изображений в метаданных
    size_elem = task.find('size')
    if size_elem is not None:
        size_elem.text = str(len(all_image_names))
    
    # Сохраняем XML файл
    xml_str = prettify(annotations)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"\nАннотация сохранена в {output_file}")
    print(f"Всего изображений: {len(all_image_names)}")
    print(f"Изображений с аннотациями: {len(image_masks)}")
    print(f"Изображений без масок: {len(all_image_names) - len(image_masks)}")

def prettify(elem):
    """Возвращает красиво отформатированную XML строку"""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

if __name__ == "__main__":
    # Укажите путь к каталогу с масками
    mask_directory = "/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno-tongue"
    
    # Укажите имя выходного файла
    output_filename = "cvat_annotations.xml"
    
    # Создаем аннотацию
    create_cvat_annotation(mask_directory, output_filename)
