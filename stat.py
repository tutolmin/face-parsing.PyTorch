import numpy as np
from model import BiSeNet
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import cv2
import os

# Конфигурация
VAL_IMAGES_DIR = "test_img/"  # Папка с валидационными изображениями
VAL_MASKS_DIR = "test_label/"    # Папка с ручными масками (классы 0-18)
#MODEL_PATH = "res/model_final_diss_16.pth"   # Путь к предобученной модели
MODEL_PATH = "res/model_final_diss.pth"   # Путь к предобученной модели
#MODEL_PATH = "res/79999_iter_orig.pth"   # Путь к предобученной модели
NUM_CLASSES = 19                # Количество классов в исходной модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели (пример для BiSeNet)
model = BiSeNet(n_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

# Список изображений для валидации
image_files = sorted(os.listdir(VAL_IMAGES_DIR))

# Метрики
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_accuracy = 0.0

# Вычисление метрик для каждого класса
class_iou = np.zeros(NUM_CLASSES)
class_counts = np.zeros(NUM_CLASSES)

with torch.no_grad():
    for img_file in tqdm(image_files, desc="Processing validation images"):
        # Загрузка изображения и маски
        img_path = os.path.join(VAL_IMAGES_DIR, img_file)
        mask_path = os.path.join(VAL_MASKS_DIR, img_file.replace(".jpg", ".png"))
        
        # Препроцессинг изображения
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0).to(DEVICE)
        
        # Загрузка ground truth маски
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        true_mask = torch.from_numpy(true_mask).long().to(DEVICE)
        
        # Предсказание модели
        pred = model(image)[0]
        pred_mask = torch.argmax(pred, dim=1).squeeze(0)
        
        # Расчет метрик
        true_mask_np = true_mask.cpu().numpy().flatten()
        pred_mask_np = pred_mask.cpu().numpy().flatten()
        
        # IoU для каждого класса
        for class_id in range(NUM_CLASSES):
            if class_id in true_mask_np:
                class_iou[class_id] += jaccard_score(
                    (true_mask_np == class_id).astype(int),
                    (pred_mask_np == class_id).astype(int),
                    zero_division=0
                )
                class_counts[class_id] += 1
        
        # Общие метрики
        total_iou += jaccard_score(true_mask_np, pred_mask_np, average="macro", zero_division=0)
        total_precision += precision_score(true_mask_np, pred_mask_np, average="macro", zero_division=0)
        total_recall += recall_score(true_mask_np, pred_mask_np, average="macro", zero_division=0)
        total_accuracy += accuracy_score(true_mask_np, pred_mask_np)

# Усреднение метрик
num_images = len(image_files)
mean_iou = total_iou / num_images
mean_precision = total_precision / num_images
mean_recall = total_recall / num_images
mean_accuracy = total_accuracy / num_images

# IoU по классам
class_iou = class_iou / (class_counts + 1e-6)  # Защита от деления на ноль

# Вывод результатов
print(f"\nValidation Metrics (on {num_images} images):")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean Accuracy: {mean_accuracy:.4f}\n")

print("Per-class IoU:")
for class_id in range(NUM_CLASSES):
    if class_counts[class_id] > 0:  # Показываем только присутствующие классы
        print(f"Class {class_id}: {class_iou[class_id]:.4f} (count: {int(class_counts[class_id])})")
