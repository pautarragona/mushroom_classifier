"""
Test dedicado para el modelo YOLO (Ultralytics clasificación)
Uso ejemplos:
    conda activate iao
    python test_yolo.py --image "Mushrooms/Agaricus/000_ePQknW8cTp8.jpg"
    python test_yolo.py --folder "imagenes_nuevas" --save
"""

import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = 'yolo/weights/best.pt'  # Ruta principal esperada (copiada manualmente)
CLASSES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.5

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics no está instalado. Ejecuta: pip install ultralytics")
    raise SystemExit(1)

def find_weights():
    # 1. Ruta principal
    main_path = Path(MODEL_PATH)
    if main_path.exists():
        return main_path
    # 2. Buscar en runs/classify/*/weights/best.pt
    runs_dir = Path('runs/classify')
    if runs_dir.exists():
        candidates = []
        for exp in runs_dir.glob('*'):
            w = exp / 'weights' / 'best.pt'
            if w.exists():
                candidates.append(w)
        if candidates:
            # Elegir el más reciente por fecha de modificación
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None

weights_path = find_weights()
if not weights_path:
    print("No se encontraron pesos entrenados de YOLO.")
    print("Debes entrenar el modelo ejecutando:")
    print("    conda activate iao")
    print("    python yolov8.py")
    print("Esto generará una carpeta en 'runs/classify/.../weights/best.pt'.")
    print("Luego puedes copiar el archivo a 'yolo/weights/best.pt' o dejar que este script lo detecte automáticamente.")
    raise SystemExit(1)

print(f"Usando pesos: {weights_path}")
model = YOLO(str(weights_path))


def predict_image(image_path: Path):
    results = model.predict(str(image_path), imgsz=IMG_SIZE, verbose=False)
    pred_idx = results[0].probs.top1
    conf = results[0].probs.top1conf.item()
    label = CLASSES[pred_idx] if conf >= UNKNOWN_THRESHOLD else 'otras'
    # top-3
    top3_indices = results[0].probs.top5[:3]
    top3 = []
    for i in top3_indices:
        prob = results[0].probs.data[i].item()
        cls_name = CLASSES[i] if prob >= UNKNOWN_THRESHOLD else 'otras'
        top3.append((cls_name, prob))
    return label, conf, top3


def show_result(image_path: Path, result, save=False):
    label, conf, top3 = result
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    color = '#2ecc71' if conf > 0.8 else '#f39c12' if conf > 0.6 else '#e74c3c'
    plt.title(f"YOLO: {label} ({conf:.1%})", color=color, fontsize=14, fontweight='bold')
    y = 0.05
    for cls, c in top3:
        plt.text(0.02, y, f"{cls}: {c:.1%}", transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        y += 0.05
    if save:
        out_dir = Path('test_results'); out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_yolo.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {out_path}")
    plt.show(); plt.close()


def process_folder(folder: Path, save=False):
    exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    images = [p for p in folder.iterdir() if p.suffix in exts]
    if not images:
        print("No hay imágenes en la carpeta."); return
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
    csv_path = out_dir / 'yolo_batch.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResumen guardado en {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Test del modelo YOLO con nuevas imágenes')
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
