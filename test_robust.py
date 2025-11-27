"""
Test dedicado para el modelo Robust (MobileNetV2 / TensorFlow)
Uso ejemplos:
    conda activate iao_tf
    python test_robust.py --image "Mushrooms/Agaricus/000_ePQknW8cTp8.jpg"
    python test_robust.py --folder "imagenes_nuevas" --save
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuración esperada
MODEL_PATH = 'imagenet/checkpoints/best_20251126_152956_phase2.h5'
CLASSES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
IMG_SIZE = 224
THRESHOLD = 0.3

os.environ['PATH'] = r'C:\Users\ASUS\.conda\envs\iao_tf\Library\bin;' + os.environ['PATH']

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow no está instalado. Activa el entorno correcto:")
    print("    conda activate iao_tf")
    raise SystemExit(1)

if not Path(MODEL_PATH).exists():
    print(f"No se encontró el modelo en {MODEL_PATH}")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo Robust cargado")

def predict_image(image_path: Path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argmax(preds)
    top_conf = preds[top_idx]
    if top_conf < THRESHOLD:
        label = 'otras'
    else:
        label = CLASSES[top_idx]
    # top-3
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = []
    for i in top3_idx:
        cls_name = CLASSES[i] if preds[i] >= THRESHOLD else 'otras'
        top3.append((cls_name, preds[i]))
    return label, float(top_conf), top3


def show_result(image_path: Path, result, save=False):
    label, conf, top3 = result
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    color = '#2ecc71' if conf > 0.8 else '#f39c12' if conf > 0.6 else '#e74c3c'
    plt.title(f"Predicción: {label} ({conf:.1%})", color=color, fontsize=14, fontweight='bold')
    y = 0.05
    for cls, c in top3:
        plt.text(0.02, y, f"{cls}: {c:.1%}", transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        y += 0.05
    if save:
        out_dir = Path('test_results')
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_robust.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Guardado: {out_path}")

    plt.show()
    plt.close()


def process_folder(folder: Path, save=False):
    exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    images = [p for p in folder.iterdir() if p.suffix in exts]
    if not images:
        print("No hay imágenes en la carpeta.")
        return
    print(f"Procesando {len(images)} imágenes...")
    import pandas as pd
    rows = []
    for img_path in images:
        label, conf, top3 = predict_image(img_path)
        rows.append({'image': img_path.name, 'prediction': label, 'confidence': conf})
        print(f"{img_path.name}: {label} ({conf:.1%})")
        if save:
            show_result(img_path, (label, conf, top3), save=True)
    df = pd.DataFrame(rows)
    out_dir = Path('test_results'); out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'robust_batch.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResumen guardado en {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Test del modelo Robust con nuevas imágenes')
    parser.add_argument('--image', type=str, help='Ruta a una imagen')
    parser.add_argument('--folder', type=str, help='Ruta a carpeta con imágenes')
    parser.add_argument('--save', action='store_true', help='Guardar visualizaciones')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help(); return

    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"Imagen no encontrada: {p}"); return
        res = predict_image(p)
        print(f"\nPredicción: {res[0]} ({res[1]:.1%})")
        print("Top-3:")
        for cls, c in res[2]:
            print(f"  - {cls}: {c:.1%}")
        show_result(p, res, save=args.save)
    else:
        f = Path(args.folder)
        if not f.exists():
            print(f"Carpeta no encontrada: {f}"); return
        process_folder(f, save=args.save)

if __name__ == '__main__':
    main()
