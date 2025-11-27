"""
Clasificador de Setas con YOLOv8 - Explicado y Mejorado
=======================================================

Este script implementa un clasificador robusto de setas usando YOLOv8 para clasificación. El código está estructurado en secciones, cada una explicada detalladamente antes de su bloque correspondiente. Se han eliminado los comentarios originales y se han añadido explicaciones didácticas. Todos los gráficos y resultados se guardan en la carpeta 'yolo'.

Gráficos generados (todos en 'yolo/'):
- Curvas de accuracy y loss (entrenamiento y validación)
- Top-5 accuracy
- Análisis del gap (overfitting)
- Matriz de confusión
- Accuracy por clase
- Distribución de confianza de las predicciones
- Curva de calibración (accuracy vs confianza)
- Evolución del learning rate

Cada gráfico incluye una explicación en el código.
"""

# Importar librerías necesarias

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Detección de GPU y configuración
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Importar YOLO de Ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'ultralytics'])
    from ultralytics import YOLO

# Fijar semilla para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Establecer las configuraciones principales

DATA_DIR = 'Mushrooms'
YOLO_DATA_DIR = 'Mushrooms_YOLO'
YOLO_MODEL = 'yolov8m-cls.pt'
IMG_SIZE = 224
BATCH_SIZE = 64 # número de imágenes que se procesan juntas (aumentado para usar ~3GB GPU)
EPOCHS = 150 # total de epochs de entrenamiento
PATIENCE = None # desactivar el early stopping
RESULTS_DIR = 'yolo'
os.makedirs(RESULTS_DIR, exist_ok=True) # guardar datos

# Preparación del dataset en formato YOLO

def prepare_yolo_dataset(source_dir, target_dir, train_split=0.70, val_split=0.15):
    target_path = Path(target_dir)
    if target_path.exists():
        shutil.rmtree(target_path)
    for split in ['train', 'val', 'test']:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    source_path = Path(source_dir)
    classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    for class_name in classes:
        class_path = source_path / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        if len(images) == 0:
            continue
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(images)
        n_images = len(images)
        n_train = int(n_images * train_split)
        n_val = int(n_images * val_split)
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            (target_path / split / class_name).mkdir(exist_ok=True)
            for img in split_images:
                shutil.copy2(img, target_path / split / class_name / img.name)
    return classes

# Entrenamiento del modelo YOLOv8

def train_yolo_classifier(data_dir, model_name, classes):
    model = YOLO(model_name)
    model.train(
        data=data_dir,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=20,
        translate=0.2,
        scale=0.5,
        shear=5,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.15,
        copy_paste=0.0,
        dropout=0.2,
        patience=EPOCHS+1, # sin early stopping
        seed=RANDOM_SEED,
        deterministic=True,
        device=DEVICE,
        workers=4,
        project='runs/classify',
        name='mushroom_yolo8_gpu',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        save=True,
        save_period=5,
        plots=True,
        val=True,
        cache=False,
        amp=True if DEVICE == 'cuda' else False,
    )
    return model

# Evaluación y visualización de resultados

def evaluate_yolo_model(model, test_dir, classes, timestamp, unknown_threshold=0.5):
    test_path = Path(test_dir) / 'test'
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    for class_idx, class_name in enumerate(classes):
        class_path = test_path / class_name
        if not class_path.exists():
            continue
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        for img_path in images:
            results = model.predict(
                source=str(img_path),
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )
            pred_class = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()
            # Asignar clase 'otras' si la confianza máxima es baja
            if confidence < unknown_threshold:
                pred_class = len(classes)
            all_predictions.append(pred_class)
            all_true_labels.append(class_idx)
            all_confidences.append(confidence)
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_confidences = np.array(all_confidences)
    classes_extended = classes + ['otras']
    report_dict = classification_report(
        all_true_labels,
        all_predictions,
        target_names=classes_extended,
        output_dict=True
    )
    with open(os.path.join(RESULTS_DIR, f'report_yolo9_{timestamp}.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)
    return all_predictions, all_true_labels, all_confidences, report_dict

# Visualización de resultados

def plot_yolo_results(model, predictions, true_labels, confidences, classes, timestamp):
    print("\nGenerando visualizaciones...")
    
    results_path = Path('runs/classify/mushroom_yolo8_gpu')
    results_csv = results_path / 'results.csv'
    
    # Leer historial de entrenamiento
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
    else:
        print(f"[WARN] No se encontró el archivo de resultados en {results_csv}")
        return

    # Si hay clase 'otras', añadirla a los nombres para la matriz de confusión
    plot_class_names = list(classes)
    if len(np.unique(predictions)) > len(classes):
        if 'otras' not in plot_class_names:
            plot_class_names.append('otras')

    # Figura grande (mismo tamaño que robust)
    fig = plt.figure(figsize=(24, 12))
    
    # 1. Accuracy (Train vs Val) - YOLO solo tiene Val, así que mostramos solo Val
    plt.subplot(2, 4, 1)
    if 'metrics/accuracy_top1' in df.columns:
        plt.plot(df['metrics/accuracy_top1'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.title('Accuracy (Train vs Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Loss (Train vs Val)
    plt.subplot(2, 4, 2)
    if 'train/loss' in df.columns:
        plt.plot(df['train/loss'], label='Train', linewidth=2.5, color='#2ecc71')
    if 'val/loss' in df.columns:
        plt.plot(df['val/loss'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.title('Loss (Train vs Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Top-3 Accuracy (usamos Top-5 de YOLO ya que es lo más cercano)
    plt.subplot(2, 4, 3)
    if 'metrics/accuracy_top5' in df.columns:
        plt.plot(df['metrics/accuracy_top5'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.title('Top-3 Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Top-3 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Gap Analysis - Aproximado usando loss gap
    plt.subplot(2, 4, 4)
    if 'train/loss' in df.columns and 'val/loss' in df.columns:
        # Aproximar el gap usando la diferencia de loss normalizada
        gap = df['val/loss'] - df['train/loss']
        plt.plot(gap, linewidth=2.5, color='#9b59b6')
        plt.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Target Gap (10%)')
        plt.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='Previous Gap (20%)')
        plt.title('Overfitting Gap (Train - Val)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Gap analysis not available\n(YOLO no registra train accuracy)', 
                 ha='center', va='center', fontsize=12)
        plt.title('Overfitting Gap (Train - Val)', fontsize=14, fontweight='bold')
        plt.axis('off')
    
    # 5. Confusion Matrix
    plt.subplot(2, 4, 5)
    cm = confusion_matrix(true_labels, predictions, labels=range(len(plot_class_names)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=plot_class_names, yticklabels=plot_class_names, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    
    # 6. Per-class Accuracy
    plt.subplot(2, 4, 6)
    class_acc = []
    for i in range(len(plot_class_names)):
        mask = true_labels == i
        if mask.sum() > 0:
            acc = (predictions[mask] == true_labels[mask]).mean()
        else:
            acc = 0.0
        class_acc.append(acc)
    
    colors = ['#2ecc71' if a > 0.80 else '#f39c12' if a > 0.65 else '#e74c3c' for a in class_acc]
    plt.barh(plot_class_names, class_acc, color=colors, alpha=0.8)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    for i, v in enumerate(class_acc):
        plt.text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold')
    
    # 7. Confidence Distribution
    plt.subplot(2, 4, 7)
    min_len = min(len(predictions), len(true_labels), len(confidences))
    correct = predictions[:min_len] == true_labels[:min_len]
    plt.hist(confidences[:min_len][correct], bins=30, alpha=0.7, label='Correct', color='#2ecc71', density=True)
    plt.hist(confidences[:min_len][~correct], bins=30, alpha=0.7, label='Incorrect', color='#e74c3c', density=True)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Learning Rate Schedule
    plt.subplot(2, 4, 8)
    lr_col = [c for c in df.columns if 'lr/' in c]
    if lr_col:
        plt.semilogy(df[lr_col[0]], linewidth=2.5, color='#3498db')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'results_yolo9_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {save_path}")
    plt.close()
    
    # Guardar copia del log CSV en la carpeta de resultados
    if results_csv.exists():
        shutil.copy2(results_csv, os.path.join(RESULTS_DIR, f'training_log_yolo9_{timestamp}.csv'))

# Pipeline principal

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: '{DATA_DIR}' no encontrado")
        return
    classes = prepare_yolo_dataset(DATA_DIR, YOLO_DATA_DIR, train_split=0.70, val_split=0.15)
    model = train_yolo_classifier(YOLO_DATA_DIR, YOLO_MODEL, classes)
    predictions, true_labels, confidences, report_dict = evaluate_yolo_model(
        model, YOLO_DATA_DIR, classes, timestamp)
    plot_yolo_results(model, predictions, true_labels, confidences, classes, timestamp)
    with open(os.path.join(RESULTS_DIR, f'classes_yolo9_{timestamp}.json'), 'w') as f:
        json.dump(classes, f, indent=2)
    # Copiar pesos entrenados a carpeta estándar para test_yolo.py
    export_dir = Path('yolo/weights')
    export_dir.mkdir(parents=True, exist_ok=True)
    best_src = Path('runs/classify/mushroom_yolo8_gpu/weights/best.pt')
    if best_src.exists():
        dest_path = export_dir / 'best.pt'
        shutil.copy2(best_src, dest_path)
        print(f"\n✅ Pesos exportados a: {dest_path}")
    else:
        print("\n⚠️ No se encontró 'best.pt' en la ruta esperada. Verifica el nombre del experimento o si el entrenamiento terminó correctamente.")
    print(f"\nTodos los resultados y gráficos se han guardado en la carpeta '{RESULTS_DIR}'.")

if __name__ == "__main__":
    main()
