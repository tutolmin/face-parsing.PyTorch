import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert CVAT annotations to CelebAMask-HQ format')
    parser.add_argument('--cvat_xml', required=True, help='Path to CVAT XML annotations file')
    parser.add_argument('--image_dir', required=True, help='Directory containing original images')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed dataset')
    parser.add_argument('--resize_img', action='store_true', help='Resize images to 1024x1024')
    parser.add_argument('--start_id', type=int, default=30000, help='Starting image ID (default: 30000)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mask'), exist_ok=True)

    # Define ONLY the classes that are present in the export
    class_mapping = {
#        'mouth': 'mouth',
#        'u_lip': 'u_lip',
#        'l_lip': 'l_lip',
        'tongue': 'tongue',
#        'occlusion': 'occlusion'
    }

    # Parse XML file
    tree = ET.parse(args.cvat_xml)
    root = tree.getroot()

    # Get all image elements
    image_elements = root.findall('image')

    print(f"Found {len(image_elements)} images to process")
    print(f"Processing classes: {list(class_mapping.keys())}")
    print(f"Starting image ID from: {args.start_id}")

    # Counter for image numbering starting from specified ID
    current_image_id = args.start_id

    for img_element in image_elements:
        image_name = img_element.get('name')
        image_name_without_ext = os.path.splitext(image_name)[0]
        image_width = int(img_element.get('width'))
        image_height = int(img_element.get('height'))

        # Generate 5-digit image ID starting from specified number
        image_id = f"{current_image_id:05d}"
        current_image_id += 1

        # Process original image
        img_path = os.path.join(args.image_dir, image_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping...")
            continue

        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Warning: Failed to load image {img_path}, skipping...")
            continue

        # Resize and save image (1024x1024)
        if args.resize_img:
            resized_img = cv2.resize(original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        else:
            resized_img = original_img

        output_img_name = f"{image_id}.jpg"
        output_img_path = os.path.join(args.output_dir, 'img', output_img_name)
        cv2.imwrite(output_img_path, resized_img)

        print(f"Processing image {image_id}: {image_name}")

        # Initialize masks ONLY for the classes we have in this export
        masks = {}
        for class_name in class_mapping.values():
            masks[class_name] = np.zeros((image_height, image_width), dtype=np.uint8)

        # Process polygons (masks)
        for polygon in img_element.findall('polygon'):
            label = polygon.get('label')
            if label not in class_mapping:
                print(f"Warning: Unknown label '{label}' found in image {image_id}, skipping...")
                continue

            points_str = polygon.get('points')
            points = []
            for point_str in points_str.split(';'):
                x, y = map(float, point_str.split(','))
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(masks[class_mapping[label]], [points], 255)

        # Process polylines (if any)
        for polyline in img_element.findall('polyline'):
            label = polyline.get('label')
            if label not in class_mapping:
                print(f"Warning: Unknown label '{label}' found in image {image_id}, skipping...")
                continue

            points_str = polyline.get('points')
            points = []
            for point_str in points_str.split(';'):
                x, y = map(float, point_str.split(','))
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            cv2.polylines(masks[class_mapping[label]], [points], isClosed=True, color=255, thickness=2)
            cv2.fillPoly(masks[class_mapping[label]], [points], 255)

        # Process points (if any) - convert to small circles
        for point in img_element.findall('points'):
            label = point.get('label')
            if label not in class_mapping:
                print(f"Warning: Unknown label '{label}' found in image {image_id}, skipping...")
                continue

            points_str = point.get('points')
            x, y = map(float, points_str.split(','))

            # Draw a small circle for point annotations
            cv2.circle(masks[class_mapping[label]], (int(x), int(y)), 5, 255, -1)

        # Save masks for each class (512x512) with 24-bit depth
        masks_saved = 0
        for class_name, mask in masks.items():
            if np.any(mask > 0):  # Only save masks that have annotations
                # Resize mask to 512x512 using nearest neighbor interpolation
#                resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

                # Convert to 24-bit RGB format
                # Create a 3-channel image where all channels have the same values
#                mask_rgb = cv2.merge([resized_mask, resized_mask, resized_mask])
                mask_rgb = cv2.merge([mask, mask, mask])

#                mask_filename = f"{image_id}_{class_name}.png"
                mask_filename = f"{image_name_without_ext}___{class_name}.png"
                mask_path = os.path.join(args.output_dir, 'mask', mask_filename)

                # Save as 24-bit PNG
                cv2.imwrite(mask_path, mask_rgb)
                masks_saved += 1

        print(f"  Saved {masks_saved} masks for image {image_id}")

    print("Conversion completed successfully!")
    print(f"Images saved to: {os.path.join(args.output_dir, 'img')}")
    print(f"Masks saved to: {os.path.join(args.output_dir, 'mask')}")
    print(f"Total images processed: {len(image_elements)}")
    print(f"Image ID range: {args.start_id:05d} - {(args.start_id + len(image_elements) - 1):05d}")
    print(f"Processed classes: {list(class_mapping.keys())}")

if __name__ == "__main__":
    main()
