import numpy as np
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score
from tqdm import tqdm
import cv2
import os
from model import BiSeNet
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys

# Проверяем наличие аргумента
if len(sys.argv) < 2:
    print("Usage: python script.py <model_path>")
    sys.exit(1)

# Configuration
#VAL_IMAGES_DIR = "test_img/"
#VAL_IMAGES_DIR = "/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img/"
VAL_IMAGES_DIR = "/home/andrei/data/CelebAMask-HQ/CelebA-HQ-eval-img_eye_g"
#VAL_MASKS_DIR = "test_label/"
#VAL_MASKS_DIR = "/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask"
#VAL_MASKS_DIR = "/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask_eye_g"
VAL_MASKS_DIR = "/home/andrei/data/CelebAMask-HQ/CelebAMask-HQ-mask_eye_g_100_full"
#MODEL_PATH = "res/cp_64/4999_iter.pth"
#MODEL_PATH = "res/cp_16/79999_iter_orig.pth"
#MODEL_PATH = "res/cp/4999_iter.pth"
MODEL_PATH = sys.argv[1]  # Первый аргумент командной строки
NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class definitions
#CLASS_NAMES = {
#    0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye',
#    6: 'nose', 7: 'mouth', 8: 'u_lip', 9: 'l_lip',
#}
CLASS_NAMES = {
        0: 'background', 1: 'skin', 2: 'brows', 3: 'eyes', 4: 'nose', 5: 'mouth', 6: 'u_lip', 7: 'l_lip', 8: 'eye_g',
}
#ignore_classes = [6, 9, 15, 16, 18]
ignore_classes = []

def process_image_batch(args):
    """Обработка батча изображений"""
    batch_files, model, device = args
    batch_metrics = {
        'iou': np.zeros(NUM_CLASSES),
        'precision': np.zeros(NUM_CLASSES),
        'recall': np.zeros(NUM_CLASSES),
        'count': np.zeros(NUM_CLASSES)
    }

    for img_file in batch_files:
        try:
            img_path = os.path.join(VAL_IMAGES_DIR, img_file)
            mask_path = os.path.join(VAL_MASKS_DIR, img_file.replace('.jpg', '.png'))

            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            with torch.no_grad():
                pred = model(image_tensor)[0]
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            for class_id in range(NUM_CLASSES):
                if class_id in true_mask:
                    y_true = (true_mask == class_id).astype(int)
                    y_pred = (pred_mask == class_id).astype(int)

                    batch_metrics['iou'][class_id] += jaccard_score(y_true, y_pred, average='micro', zero_division=0)
                    batch_metrics['precision'][class_id] += precision_score(y_true, y_pred, average='micro', zero_division=0)
                    batch_metrics['recall'][class_id] += recall_score(y_true, y_pred, average='micro', zero_division=0)
                    batch_metrics['count'][class_id] += 1

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    return batch_metrics

# Load model
print("Loading model " + MODEL_PATH)
model = BiSeNet(n_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Initialize metrics
metrics = {
    'iou': np.zeros(NUM_CLASSES),
    'precision': np.zeros(NUM_CLASSES),
    'recall': np.zeros(NUM_CLASSES),
    'count': np.zeros(NUM_CLASSES)
}

# Main validation loop with parallel processing
image_files = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
image_files = image_files[:1024]

batch_size = 16
num_threads = 4
batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]

print(f"Processing {len(image_files)} images in {len(batches)} batches with {num_threads} threads...")

metrics_lock = Lock()

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_image_batch, (batch, model, DEVICE)) for batch in batches]

    for future in tqdm(as_completed(futures), total=len(batches)):
        batch_metrics = future.result()
        with metrics_lock:
            for key in metrics:
                metrics[key] += batch_metrics[key]

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
        print(f"{class_id}\t{CLASS_NAMES[class_id]:<10}\t{iou:.3f}\t{prec:.3f}\t{rec:.3f}\t{int(metrics['count'][class_id])}")

valid_classes = [c for c in range(NUM_CLASSES) if c not in ignore_classes and metrics['count'][c] > 0]
mean_iou = np.mean([metrics['iou'][c]/metrics['count'][c] for c in valid_classes])
print(f"\nMean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {np.mean([metrics['precision'][c]/metrics['count'][c] for c in valid_classes]):.4f}")
print(f"Mean Recall: {np.mean([metrics['recall'][c]/metrics['count'][c] for c in valid_classes]):.4f}")
