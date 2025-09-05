import numpy as np
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt
import sys
from model import BiSeNet  # Ensure this import works from face-parsing.PyTorch

# Configuration
VAL_IMAGES_DIR = "test_img/"  # Папка с валидационными изображениями
VAL_MASKS_DIR = "test_label/"    # Папка с ручными масками (классы 0-18)
#MODEL_PATH = "res/model_final_diss_16.pth"   # Путь к предобученной модел
#MODEL_PATH = "res/model_final_diss_96.pth"   # Путь к предобученной модел
#MODEL_PATH = "res/cp/99999_iter.pth"   # Путь к предобученной модел
MODEL_PATH = "res/cp/99999_iter.pth"   # Путь к предобученной модел
#VAL_IMAGES_DIR = "val_images/"
#VAL_MASKS_DIR = "val_masks/"
#MODEL_PATH = "pretrained.pth"
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class definitions (CelebAMask-HQ standard)
CLASS_NAMES = {
    0: 'background',
    1: 'skin',
    2: 'l_brow',
    3: 'r_brow',
    4: 'l_eye',
    5: 'r_eye',
#    6: 'eye_g',
#    7: 'l_ear',
#    8: 'r_ear',
#    9: 'ear_r',
    6: 'nose',
    7: 'mouth',
    8: 'u_lip',
    9: 'l_lip',
#    14: 'neck',
#    15: 'neck_l',
#    16: 'cloth',
#    17: 'hair',
#    18: 'hat'
}
#ignore_classes = [6, 9, 15, 16, 18]          # Классы, которые нужно пропустить
ignore_classes = []          # Классы, которые нужно пропустить

def validate_mask_classes(masks_dir):
    """Check for unexpected class IDs in masks"""
    class_ids = set()
    for mask_file in os.listdir(masks_dir)[:100]:  # Check first 100 masks
        mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        unique = np.unique(mask)
        class_ids.update(unique)
        if max(unique) > 18:
            print(f"⚠️ Unexpected class {max(unique)} in {mask_file}")
    return sorted(class_ids)

# Load model
print("Loading model " + MODEL_PATH)
model = BiSeNet(n_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Class verification
print("Validating mask classes...")
detected_classes = validate_mask_classes(VAL_MASKS_DIR)
print(f"Detected classes: {detected_classes}")

# Initialize metrics
metrics = {
    'iou': np.zeros(NUM_CLASSES),
    'precision': np.zeros(NUM_CLASSES),
    'recall': np.zeros(NUM_CLASSES),
    'count': np.zeros(NUM_CLASSES)
}

# Main validation loop
image_files = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])

for i, img_file in enumerate(tqdm(image_files[:1000])):  # Limit to first 1000 for testing
    # Load and preprocess image
    img_path = os.path.join(VAL_IMAGES_DIR, img_file)
    mask_path = os.path.join(VAL_MASKS_DIR, img_file.replace('.jpg', '.png'))
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    
    # Load and verify mask
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    true_mask = cv2.resize(true_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Prediction
    with torch.no_grad():
        pred = model(image_tensor)[0]
    pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    
    # Visual check for first 3 samples
    if i < 3:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(image); plt.title("Input")
        plt.subplot(1,3,2); plt.imshow(true_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES); plt.title("GT Mask")
        plt.subplot(1,3,3); plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES); plt.title("Pred Mask")
        plt.show()
    
    # Calculate metrics per class
    for class_id in range(NUM_CLASSES):
        if class_id in true_mask:  # Only evaluate present classes
            y_true = (true_mask == class_id).astype(int)
            y_pred = (pred_mask == class_id).astype(int)
            
            metrics['iou'][class_id] += jaccard_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['precision'][class_id] += precision_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['recall'][class_id] += recall_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['count'][class_id] += 1

# Calculate final metrics
print("\nClass-wise Metrics:")
print("Class\tName\t\tIoU\tPrec\tRecall\tCount")
for class_id in range(NUM_CLASSES):
    if class_id in ignore_classes:
        continue  # Пропускаем ненужные классы    
    if metrics['count'][class_id] > 0:
        iou = metrics['iou'][class_id] / metrics['count'][class_id]
        prec = metrics['precision'][class_id] / metrics['count'][class_id]
        rec = metrics['recall'][class_id] / metrics['count'][class_id]
        
        print(f"{class_id}\t{CLASS_NAMES[class_id]:<10}\t{iou:.3f}\t{prec:.3f}\t{rec:.3f}\t{int(metrics['count'][class_id])}")

# Aggregate metrics# Сначала отфильтруем классы
valid_classes = [
    c for c in range(NUM_CLASSES) 
    if c not in ignore_classes and metrics['count'][c] > 0
]
mean_iou = np.mean([metrics['iou'][c]/metrics['count'][c] for c in valid_classes])
print(f"\nMean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {np.mean(metrics['precision'][metrics['count']>0]/metrics['count'][metrics['count']>0]):.4f}")
print(f"Mean Recall: {np.mean(metrics['recall'][metrics['count']>0]/metrics['count'][metrics['count']>0]):.4f}")
