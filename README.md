# 🍄 Clasificador de Setas con Deep Learning

Proyecto de clasificación de setas usando dos arquitecturas de redes neuronales: **MobileNetV2** (Transfer Learning con ImageNet) y **YOLOv8** (modelo de clasificación).

## 📋 Descripción

Este proyecto implementa y compara dos enfoques para clasificar especies de setas:

1. **MobileNetV2 + Transfer Learning (TensorFlow/Keras)**
   - Preentrenado en ImageNet
   - Fine-tuning en dos fases
   - Data augmentation agresivo (Mixup, Cutout, rotaciones, flips)
   - Test-Time Augmentation (TTA)
   - Regularización extensiva (Dropout, BatchNorm, L2)

2. **YOLOv8 Classification (PyTorch)**
   - YOLOv8m-cls con arquitectura optimizada
   - Data augmentation nativo de YOLO
   - Inference rápida
   - Entrenamiento con Mixup y técnicas modernas

## 🚀 Características

- ✅ **Entrenamiento en GPU** (CUDA) con aceleración automática
- ✅ **Data augmentation agresivo**: rotación, flip, crop, cutout, mixup, cambios de color
- ✅ **Detección de clase "otras"**: threshold de confianza para setas desconocidas
- ✅ **Visualización completa**: 8 gráficos de análisis (accuracy, loss, confusion matrix, etc.)
- ✅ **Métricas detalladas**: precision, recall, F1-score por clase
- ✅ **Class weights**: balanceo automático de clases desbalanceadas
- ✅ **Two-phase training**: entrenamiento progresivo para evitar overfitting

## 📂 Estructura del Proyecto

```
Práctica Final v3/
├── mushroom_classifier_robust_explicado.py  # Modelo MobileNetV2 (TensorFlow)
├── mushroom_classifier_yolo9_explicado.py   # Modelo YOLOv8 (PyTorch)
├── imagenet/                                # Resultados MobileNetV2
│   ├── results_robust_*.png                 # Visualizaciones completas
│   ├── report_robust_*.json                 # Métricas detalladas
│   ├── history_robust_*.json                # Historial de entrenamiento
│   ├── training_log_*.csv                   # Logs de entrenamiento
│   ├── mushroom_robust_*.h5                 # Modelo guardado
│   └── checkpoints/                         # Best checkpoints
├── yolo/                                    # Resultados YOLOv8
│   ├── results_yolo9_*.png                  # Visualizaciones completas
│   ├── report_yolo9_*.json                  # Métricas detalladas
│   ├── training_log_yolo9_*.csv             # Logs de entrenamiento
│   └── classes_yolo9_*.json                 # Mapeo de clases
├── .gitignore                               # Archivos excluidos de Git
└── README.md                                # Este archivo
```

## 🛠️ Requisitos e Instalación

### Entorno para MobileNetV2 (TensorFlow con GPU):
```bash
# Crear entorno con CUDA 11.2 incluido
conda create -n iao_tf python=3.9 cudatoolkit=11.2 cudnn=8.1 -c conda-forge -y

# Activar entorno
conda activate iao_tf

# Instalar dependencias
pip install tensorflow==2.10.1
pip install "numpy<2" pillow scikit-learn matplotlib seaborn
```

### Entorno para YOLOv8 (PyTorch con GPU):
```bash
# Crear entorno
conda create -n iao python=3.8 -y

# Activar entorno
conda activate iao

# Instalar PyTorch con CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias
pip install ultralytics scikit-learn matplotlib seaborn pillow
```

### Dataset:
- Coloca las imágenes de setas en la carpeta `Mushrooms/`
- Cada clase debe estar en su propia subcarpeta
- Ejemplo: `Mushrooms/Agaricus/`, `Mushrooms/Amanita/`, etc.

## 📊 Uso

### Entrenar MobileNetV2:
```bash
conda activate iao_tf
python mushroom_classifier_robust_explicado.py
```

**Configuración:**
- Batch size: 48
- Epochs: 150 (75 fase 1 + 75 fase 2)
- Image size: 224x224
- GPU memory: ~3GB

### Entrenar YOLOv8:
```bash
conda activate iao
python mushroom_classifier_yolo9_explicado.py
```

**Configuración:**
- Batch size: 64
- Epochs: 150
- Image size: 224x224
- GPU memory: ~3GB

## 📈 Resultados y Visualizaciones

Ambos modelos generan automáticamente:

1. **Accuracy Curves**: Train vs Validation accuracy por época
2. **Loss Curves**: Train vs Validation loss por época
3. **Top-3 Accuracy**: Precisión considerando las 3 predicciones más probables
4. **Overfitting Gap Analysis**: Diferencia entre train y val accuracy
5. **Confusion Matrix**: Matriz de confusión con todas las clases
6. **Per-Class Accuracy**: Accuracy individual por cada especie
7. **Confidence Distribution**: Distribución de confianza en predicciones correctas/incorrectas
8. **Learning Rate Schedule**: Evolución del learning rate durante entrenamiento

Además:
- Reportes JSON con precision, recall, F1-score por clase
- Logs CSV con todas las métricas por época
- Modelos guardados (.h5 o .pt)

## 🎯 Técnicas Implementadas

### MobileNetV2 (Robust):
- **Transfer Learning**: Pesos preentrenados de ImageNet
- **Two-phase training**: 
  - Fase 1: Solo entrenar cabeza clasificadora
  - Fase 2: Fine-tuning de últimas 80 capas del backbone
- **Regularización**: Dropout (0.4-0.6), L2, BatchNorm, GaussianNoise, SpatialDropout2D
- **Data Augmentation**: Mixup, rotación, flip, crop, cutout, HSV, contrast
- **Label Smoothing**: 0.1
- **Class Weights**: Balanceo automático
- **ReduceLROnPlateau**: Reducción adaptativa del learning rate
- **Test-Time Augmentation**: 5 predicciones promediadas

### YOLOv8:
- **YOLOv8m-cls**: Modelo medio optimizado para clasificación
- **Data Augmentation nativo**: Mixup, HSV, rotación, flip, scale, translate
- **AdamW optimizer**: Mejor convergencia
- **Learning rate schedule**: Cosine annealing
- **Dropout**: 0.2
- **AMP (Automatic Mixed Precision)**: Entrenamiento más rápido en GPU

## 🔧 Configuración Avanzada

Puedes ajustar los parámetros principales en cada script:

```python
# mushroom_classifier_robust_explicado.py
BATCH_SIZE = 48          # Tamaño de batch
EPOCHS_PHASE1 = 75       # Epochs fase 1
EPOCHS_PHASE2 = 75       # Epochs fase 2
IMG_SIZE = (224, 224)    # Tamaño de imagen
MIXUP_ALPHA = 0.2        # Intensidad de Mixup

# mushroom_classifier_yolo9_explicado.py
BATCH_SIZE = 64          # Tamaño de batch
EPOCHS = 150             # Total epochs
IMG_SIZE = 224           # Tamaño de imagen
```

## 📝 Notas Importantes

### GPU y CUDA:
- **TensorFlow 2.10** requiere CUDA 11.x (usar entorno `iao_tf` con conda)
- **PyTorch** funciona con CUDA 12.x (usar entorno `iao`)
- Ambos modelos usan ~3GB de VRAM
- Si no hay GPU, entrenarán en CPU (mucho más lento)

### Clase "otras":
- Ambos modelos detectan imágenes con baja confianza (< 50%)
- Se asignan a la clase "otras" para mejorar robustez
- Útil para detectar setas fuera de las clases entrenadas

### Compatibilidad:
- Los gráficos generados por ambos modelos tienen el **mismo formato**
- Facilita la comparación directa de resultados
- Mismo split de datos (70% train, 15% val, 15% test)

## 👥 Autores

Proyecto desarrollado para la asignatura de **Inteligencia Artificial y Optimización (IAO)** - UC3M

## 📄 Licencia

Este proyecto es de uso académico.

## 🤝 Contribuciones

Para reportar problemas o sugerencias, abre un issue en el repositorio.

---

**¡Buena suerte con la clasificación de setas! 🍄🤖**
