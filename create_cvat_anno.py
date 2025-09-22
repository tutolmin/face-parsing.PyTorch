import os
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import glob

def create_simple_cvat_annotation(mask_dir, output_file):
    """
    Упрощенная версия создания аннотации CVAT 1.1
    """
    
    classes = ['skin', 'brows', 'eyes', 'eye_g', 'nose', 'mouth', 'u_lip', 'l_lip']
    
    annotations = Element('annotations')
    version = SubElement(annotations, 'version')
    version.text = '1.1'
    
    # Собираем все файлы масок
    mask_files = glob.glob(os.path.join(mask_dir, '*_*.png'))
    
    # Группируем маски по имени изображения
    image_masks = {}
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        parts = filename.split('_')
        if len(parts) >= 3:
            image_name = f"{parts[0]}_{parts[1]}"
            class_part = '_'.join(parts[2:]).replace('.png', '')
            
            if class_part in classes:
                if image_name not in image_masks:
                    image_masks[image_name] = []
                image_masks[image_name].append((class_part, mask_file))
    
    image_id = 0
    
    for image_name, masks in image_masks.items():
        image_filename = f"{image_name}.png"
        
        image_elem = SubElement(annotations, 'image')
        image_elem.set('id', str(image_id))
        image_elem.set('name', image_filename)
        image_elem.set('width', '1024')
        image_elem.set('height', '1024')
        
        for class_name, mask_path in masks:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                continue
            
            # Используем упрощенный подход для нахождения контуров
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                polygon = SubElement(image_elem, 'polygon')
                polygon.set('label', class_name)
                polygon.set('source', 'semi-auto')
                polygon.set('occluded', '0')
                polygon.set('z_order', '0')
                
                points_str = ""
                for point in contour:
                    x, y = point[0]
                    points_str += f"{x:.2f},{y:.2f};"
                
                points_str = points_str[:-1]
                polygon.set('points', points_str)
        
        image_id += 1
    
    # Сохраняем файл
    xml_str = tostring(annotations, 'utf-8')
    reparsed = minidom.parseString(xml_str)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"Аннотация сохранена в {output_file}")
    print(f"Обработано изображений: {len(image_masks)}")

if __name__ == "__main__":
    mask_directory = "CelebAMask-HQ-mask-anno-tongue"
    output_filename = "cvat_annotations.xml"
    
    create_simple_cvat_annotation(mask_directory, output_filename)

