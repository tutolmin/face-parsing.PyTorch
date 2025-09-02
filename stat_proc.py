import numpy as np
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score
from tqdm import tqdm
import cv2
import os
from model import BiSeNet
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Должно быть в самом начале, до любых импортов torch/cuda
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Configuration
VAL_IMAGES_DIR = "test_img/"
VAL_MASKS_DIR = "test_label/"
MODEL_PATH = "res/cp/19999_iter.pth"
NUM_CLASSES = 19
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class definitions
CLASS_NAMES = {
    0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye',
    7: 'l_ear', 8: 'r_ear', 10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip',
    14: 'neck', 17: 'hair',
}
ignore_classes = [6, 9, 15, 16, 18]

def process_single_image(args):
    """Обработка одного изображения"""
    img_file, model_path, device_str = args
    try:
        # Локальная загрузка модели в каждом процессе
        local_model = BiSeNet(n_classes=NUM_CLASSES)
        local_model.load_state_dict(torch.load(model_path, map_location=device_str))

        # Для CPU устройств
        if device_str == "cuda" and torch.cuda.is_available():
            local_model.cuda()
        else:
            local_model.cpu()

        local_model.eval()

        img_path = os.path.join(VAL_IMAGES_DIR, img_file)
        mask_path = os.path.join(VAL_MASKS_DIR, img_file.replace('.jpg', '.png'))

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        # Создаем tensor на правильном устройстве
        if device_str == "cuda" and torch.cuda.is_available():
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cpu() / 255.0

        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if true_mask is None:
            return None
        true_mask = cv2.resize(true_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        with torch.no_grad():
            pred = local_model(image_tensor)[0]
        pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        image_metrics = {
            'iou': np.zeros(NUM_CLASSES),
            'precision': np.zeros(NUM_CLASSES),
            'recall': np.zeros(NUM_CLASSES),
            'count': np.zeros(NUM_CLASSES)
        }

        for class_id in range(NUM_CLASSES):
            if np.any(true_mask == class_id):
                y_true = (true_mask == class_id).astype(int).flatten()
                y_pred = (pred_mask == class_id).astype(int).flatten()

                if len(y_true) > 0 and len(y_pred) > 0:
                    image_metrics['iou'][class_id] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
                    image_metrics['precision'][class_id] = precision_score(y_true, y_pred, average='micro', zero_division=0)
                    image_metrics['recall'][class_id] = recall_score(y_true, y_pred, average='micro', zero_division=0)
                    image_metrics['count'][class_id] = 1

        return image_metrics

    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        return None

def main():
    # Main validation loop with parallel processing
    image_files = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    image_files = image_files[:1024]

    num_processes = min(mp.cpu_count(), 16)  # Ограничиваем для стабильности
    print(f"Processing {len(image_files)} images with {num_processes} processes...")

    metrics = {
        'iou': np.zeros(NUM_CLASSES),
        'precision': np.zeros(NUM_CLASSES),
        'recall': np.zeros(NUM_CLASSES),
        'count': np.zeros(NUM_CLASSES)
    }

    # Подготовка аргументов для каждого процесса
    tasks = [(img_file, MODEL_PATH, DEVICE) for img_file in image_files]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(image_files)):
            result = future.result()
            if result is not None:
                for key in metrics:
                    metrics[key] += result[key]

    # Calculate final metrics
    print("\nClass-wise Metrics:")
    print("Class\tName\t\tIoU\tPrec\tRecall\tCount")
    for class_id in range(NUM_CLASSES):
        if class_id in ignore_classes:
            continue
        if metrics['count'][class_id] > 0:
            iou = metrics['iou'][class_id] / metrics['count'][class_id]
            prec = metrics['precision'][class_id] / metrics['count'][class_id]
            rec = metrics['recall'][class_id] / metrics['count'][class_id]
            print(f"{class_id}\t{CLASS_NAMES.get(class_id, 'unknown'):<10}\t{iou:.3f}\t{prec:.3f}\t{rec:.3f}\t{int(metrics['count'][class_id])}")

    valid_classes = [c for c in range(NUM_CLASSES) if c not in ignore_classes and metrics['count'][c] > 0]
    if valid_classes:
        mean_iou = np.mean([metrics['iou'][c]/metrics['count'][c] for c in valid_classes])
        mean_precision = np.mean([metrics['precision'][c]/metrics['count'][c] for c in valid_classes])
        mean_recall = np.mean([metrics['recall'][c]/metrics['count'][c] for c in valid_classes])
        print(f"\nMean IoU: {mean_iou:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")

if __name__ == "__main__":
    main()
