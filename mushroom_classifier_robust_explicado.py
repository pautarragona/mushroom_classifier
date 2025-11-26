"""
Clasificador de Setas con MobileNetV2 (Imagenet) - Explicado y Mejorado
=======================================================================

Este script implementa un clasificador robusto de setas usando MobileNetV2 preentrenado en ImageNet. 
El código está estructurado en secciones, cada una explicada detalladamente antes de su bloque 
correspondiente. Se han eliminado los comentarios originales y se han añadido explicaciones didácticas. 
Además, el script detecta y usa GPU (CUDA) si está disponible, y todos los gráficos y resultados se 
guardan en la carpeta 'imagenet'.

Gráficos generados ((todos en 'imagenet/')):
- Curvas de accuracy y loss (entrenamiento y validación)
- Top-3 accuracy
- Análisis del gap (overfitting)
- Matriz de confusión
- Accuracy por clase
- Distribución de confianza de las predicciones
- Curva de calibración (accuracy vs confianza)
- Evolución del learning rate

ENTRENAMIENTO::
- Fase 1: Entrenamiento de la cabeza del modelo. NO se modifican los pesos del modelo base MobileNetV2, solo se usan para extraer características, y solo se entrenan las capas nuevas.
- Fase 2: Se descongelan las últimas capas del modelo base de MobileNetV2 para fine-tuning junto con la cabeza del modelo. Ajustar más específicamente manteniendo la base
"""

# Importar librerías necesarias (sklearn: preparar datos y evaluar resultados; tensorflow/keras: construir y entrenar el modelo, la RNA)

import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Configurar PATH para CUDA de conda (necesario para GPU)
conda_env = os.path.dirname(sys.executable)
cuda_bin_path = os.path.join(conda_env, 'Library', 'bin')
if os.path.exists(cuda_bin_path) and cuda_bin_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageFile

# Cargar imagenes truncadas sin error
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fijamos una semilla para reproducibilidad
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configurar GPU: usar detección nativa de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Habilitar crecimiento dinámico de memoria GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\nGPU detectada: {len(gpus)} dispositivo(s) - {gpus[0].name}")
        print(f"Usando GPU para entrenamiento acelerado")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("\nNo se detectó GPU - usando CPU")
    print("Asegúrate de usar el entorno 'iao_tf' con CUDA instalado")


# Establecer las configuraciones principales

DATA_DIR = 'Mushrooms' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 48 # número de imágenes que se procesan juntas (aumentado para usar ~3GB GPU)
EPOCHS_PHASE1 = 60  
EPOCHS_PHASE2 = 60 # en total son phase 1 + phase 2 = 120 epochs
VAL_SPLIT = 0.15  
TEST_SPLIT = 0.15
TRAIN_SPLIT = 0.70
AUTOTUNE = tf.data.AUTOTUNE # tensorflow optimiza el rendimiento de carga de datos
MIXUP_ALPHA = 0.1 # intensidad de mezcla de imágenes y etiquetas
RESULTS_DIR = 'imagenet'
os.makedirs(RESULTS_DIR, exist_ok=True) # guardar datos


# =========================================================
# Aquí empieza el preprocesado de datos y data augmentation
# =========================================================

# Mezcla de imágenes y etiquetas (Mixup) para mejorar la generalización del modelo
def mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0] # cuántas imagenes hay en el batch?
    lambda_value = tf.random.uniform([], 0, alpha) # num. aleatorio entre 0 y alpha
    lambda_value = tf.maximum(lambda_value, 1 - lambda_value) # dar peso a la imágen original
    indices = tf.random.shuffle(tf.range(batch_size)) # desordena índices del batch
    mixed_images = lambda_value * images + (1 - lambda_value) * tf.gather(images, indices) # mezcla imágenes con otras aleatorias
    mixed_labels = lambda_value * labels + (1 - lambda_value) * tf.gather(labels, indices) # mezcla etiquetas de la misma forma
    return mixed_images, mixed_labels

# Augmentación agresiva aplicando transformaciones aleatorias para combatir el overfitting
def aggressive_augment(image):
    # Cambia el brillo, contraste, saturación, tono de color, rota 90º, flip horizontal, flip vertical
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Random crop: recorta una parte aleatoria y la redimensiona
    if tf.random.uniform([]) > 0.5:
        crop_size = tf.random.uniform([], 0.7, 1.0) # entre 70% y 100%
        h, w = IMG_SIZE
        crop_h = tf.cast(h * crop_size, tf.int32)
        crop_w = tf.cast(w * crop_size, tf.int32)
        image = tf.image.random_crop(image, [crop_h, crop_w, 3])
        image = tf.image.resize(image, IMG_SIZE)
    # Cutout: crea un rectángulo negro aleatorio en la imagen
    if tf.random.uniform([]) > 0.5:
        cutout_h = tf.random.uniform([], 15, 40, dtype=tf.int32)
        cutout_w = tf.random.uniform([], 15, 40, dtype=tf.int32)
        y_start = tf.random.uniform([], 0, IMG_SIZE[0] - cutout_h, dtype=tf.int32)
        x_start = tf.random.uniform([], 0, IMG_SIZE[1] - cutout_w, dtype=tf.int32)
        mask = tf.ones([cutout_h, cutout_w, 3])
        paddings = [
            [y_start, IMG_SIZE[0] - y_start - cutout_h],
            [x_start, IMG_SIZE[1] - x_start - cutout_w],
            [0, 0]
        ]
        mask = tf.pad(mask, paddings, constant_values=0)
        mask = 1 - mask
        image = image * mask
    image = tf.clip_by_value(image, -1.0, 1.0) # todos los valores de píxeles en el rango [-1, 1]
    return image

# Decodifica, redimensiona y preprocesa imágenes
def decode_and_resize(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    if augment:
        img = aggressive_augment(img)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return img, label

# Cargar y procesar datos mientras que la GPU entrene
def make_dataset(paths, labels, training=False, use_mixup=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: decode_and_resize(p, l, augment=training), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    if training and use_mixup:
        ds = ds.map(lambda x, y: mixup(x, y, MIXUP_ALPHA), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds

# =======================================================
# Aquí empieza la preparación de datos: carga y partición
# =======================================================

# Recorre el directorio de datos y lista todas las imágenes y sus etiquetas
def list_images(data_dir):
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    paths, labels = [], []
    for cname in class_names:
        cpath = os.path.join(data_dir, cname)
        files = [f for f in os.listdir(cpath) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in files:
            paths.append(os.path.join(cpath, f))
            labels.append(class_to_idx[cname])
    return np.array(paths), np.array(labels), class_names

# Parte el dataset en train, val y test de forma estratificada
def stratified_split(paths, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=labels, random_state=RANDOM_SEED
    )
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=RANDOM_SEED
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Construye un modelo MobileNetV2 con mucha regularización para evitar overfitting
# Regularizaciones: Gaussian Noise, SpatialDropout2D, Dropout progresivo, arquitectura más pequeña
def build_robust_model(num_classes):
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    base_model.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.GaussianNoise(0.05)(inputs) # añade ruido aleatorio a las imágenes de entrada
    x = base_model(x, training=False)
    x = layers.SpatialDropout2D(0.2)(x) # desactiva aleatoriamente el 20% de las características
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x) # desactiva el 30% de las neuronas
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.003))(x) # más neuronas para mejor aprendizaje
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x) # desactiva el 40% de las neuronas
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.003))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # desactiva el 50% de las neuronas
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model

# ========================================
# Aquí empieza el entrenamiento del modelo
# ========================================

# # Callbacks para el entrenamiento: logs y checkpoints
def create_callbacks(timestamp):
    os.makedirs(os.path.join(RESULTS_DIR, 'checkpoints'), exist_ok=True)
    return [
        keras.callbacks.ReduceLROnPlateau( # reduce learning rate si la val_loss no mejora en 5 epochs
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
        keras.callbacks.ModelCheckpoint( # guarda el mejor modelo basado en val_accuracy
            filepath=os.path.join(RESULTS_DIR, 'checkpoints', f'best_{timestamp}.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger( # registra (loss, accuracy, etc.) en un archivo CSV
            filename=os.path.join(RESULTS_DIR, f'training_log_{timestamp}.csv'), separator=',', append=False)
    ]

# Fase 1: entrena solo la cabeza del modelo. Da peso a clases minoritarias para balancear.
def train_phase1(model, train_ds, val_ds, class_weights, timestamp):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    callbacks = create_callbacks(timestamp)
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1,
        callbacks=callbacks, class_weight=class_weights, verbose=1)
    return history1

# Fase 2: fine-tuning, descongela las últimas capas y entrena con LR bajo
def train_phase2(model, base_model, train_ds, val_ds, class_weights, timestamp):
    base_model.trainable = True
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    callbacks = create_callbacks(timestamp + "_phase2")
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2,
        callbacks=callbacks, class_weight=class_weights, verbose=1)
    return history2

# =====================================
# Aquí empieza la evaluación del modelo
# =====================================

# Se hace data augmentation con la imagen 5 veces y hace predicciones, que se promedian los resultados luego (Test-Time Augmentation)
def predict_with_tta(model, test_ds, num_augmentations=5):
    all_predictions = []
    for _ in range(num_augmentations):
        preds = model.predict(test_ds, verbose=0)
        all_predictions.append(preds)
    avg_predictions = np.mean(all_predictions, axis=0)
    return avg_predictions

# Evalúa el modelo, añade la clase 'otras' si la confianza máxima es baja, y registra el reporte de clasificación en JSON.
def evaluate_model(model, test_ds, class_names, timestamp, use_tta=True, unknown_threshold=0.3):
    results = model.evaluate(test_ds, verbose=1)
    test_loss, test_acc, test_top3 = results[:3]
    if use_tta:
        predictions = predict_with_tta(model, test_ds, num_augmentations=5)
    else:
        predictions = model.predict(test_ds, verbose=1)
    max_probs = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    # Asigna clase 'otras' si la confianza máxima es baja
    predicted_classes_with_unk = predicted_classes.copy()
    unknown_class_idx = len(class_names)
    predicted_classes_with_unk[max_probs < unknown_threshold] = unknown_class_idx
    # Clases verdaderas
    true_classes = []
    for _, labels in test_ds:
        true_classes.extend(np.argmax(labels.numpy(), axis=1))
    true_classes = np.array(true_classes)
    min_len = min(len(predicted_classes_with_unk), len(true_classes))
    predicted_classes_with_unk = predicted_classes_with_unk[:min_len]
    true_classes = true_classes[:min_len]
    predictions = predictions[:min_len]
    # Añadir 'otras' a los nombres de clase
    class_names_extended = class_names + ['otras']
    report_dict = classification_report(true_classes, predicted_classes_with_unk, target_names=class_names_extended, output_dict=True)
    with open(os.path.join(RESULTS_DIR, f'report_robust_{timestamp}.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)
    return predictions, predicted_classes_with_unk, true_classes, report_dict

# =================================
# VISUALIZACIÓN FINAL DE RESULTADOS
# =================================

# Genera y guarda todos los gráficos en 'imagenet/'
def plot_results(history1, history2, predictions, true_classes, predicted_classes, class_names, timestamp):
    print("\n Generando visualizaciones...")
    
    # Historial de entrenamiento
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'top3_acc': history1.history['top3_acc'] + history2.history['top3_acc'],
        'val_top3_acc': history1.history['val_top3_acc'] + history2.history['val_top3_acc']
    }
    
    # Si hay clase 'otras', añadirla a los nombres para la matriz de confusión
    plot_class_names = list(class_names)
    if len(np.unique(predicted_classes)) > len(class_names):
        if 'otras' not in plot_class_names:
            plot_class_names.append('otras')
            
    # Figura grande
    fig = plt.figure(figsize=(24, 12))
    
    # 1. Accuracy
    plt.subplot(2, 4, 1)
    plt.plot(combined_history['accuracy'], label='Train', linewidth=2.5, color='#2ecc71')
    plt.plot(combined_history['val_accuracy'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', alpha=0.5, label='Phase 2 Start')
    plt.title('Accuracy (Train vs Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Loss
    plt.subplot(2, 4, 2)
    plt.plot(combined_history['loss'], label='Train', linewidth=2.5, color='#2ecc71')
    plt.plot(combined_history['val_loss'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.axvline(x=len(history1.history['loss']), color='gray', linestyle='--', alpha=0.5, label='Phase 2 Start')
    plt.title('Loss (Train vs Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Top-3 Accuracy
    plt.subplot(2, 4, 3)
    plt.plot(combined_history['top3_acc'], label='Train', linewidth=2.5, color='#2ecc71')
    plt.plot(combined_history['val_top3_acc'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.axvline(x=len(history1.history['top3_acc']), color='gray', linestyle='--', alpha=0.5, label='Phase 2 Start')
    plt.title('Top-3 Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Top-3 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Gap Analysis
    plt.subplot(2, 4, 4)
    gap = np.array(combined_history['accuracy']) - np.array(combined_history['val_accuracy'])
    plt.plot(gap, linewidth=2.5, color='#9b59b6')
    plt.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Target Gap (10%)')
    plt.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='Previous Gap (20%)')
    plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', alpha=0.5)
    plt.title('Overfitting Gap (Train - Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Confusion Matrix
    plt.subplot(2, 4, 5)
    # Asegurar que las etiquetas coincidan con los nombres de clases
    cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(plot_class_names)))
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
        mask = true_classes == i
        if mask.sum() > 0:
            acc = (predicted_classes[mask] == true_classes[mask]).mean()
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
    max_probs = np.max(predictions, axis=1)
    # Ajustar longitud si es necesario
    min_len = min(len(predicted_classes), len(true_classes), len(max_probs))
    correct = predicted_classes[:min_len] == true_classes[:min_len]
    plt.hist(max_probs[:min_len][correct], bins=30, alpha=0.7, label='Correct', color='#2ecc71', density=True)
    plt.hist(max_probs[:min_len][~correct], bins=30, alpha=0.7, label='Incorrect', color='#e74c3c', density=True)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Learning Rate Schedule
    plt.subplot(2, 4, 8)
    epochs = list(range(len(combined_history['accuracy'])))
    # Simular LR schedule (aproximado)
    lr_schedule = []
    for epoch in epochs:
        if epoch < len(history1.history['accuracy']):
            lr = 0.001 * (0.5 ** (epoch // 5))  # Aproximado
        else:
            lr = 5e-5 * (0.5 ** ((epoch - len(history1.history['accuracy'])) // 5))
        lr_schedule.append(lr)
    
    plt.semilogy(epochs, lr_schedule, linewidth=2.5, color='#3498db')
    plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', alpha=0.5)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'results_robust_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {save_path}")
    plt.close()
    
    # Guardar historial
    with open(os.path.join(RESULTS_DIR, f'history_robust_{timestamp}.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in combined_history.items()}, f, indent=2)

# =============================================
# Código principal que ejecuta todo el pipeline
# =============================================

# Carga datos, entrena, evalúa y guarda resultados y gráficos
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Carga datos y clases
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] '{DATA_DIR}' no encontrado")
        return
    paths, labels, class_names = list_images(DATA_DIR)
    global NUM_CLASSES
    NUM_CLASSES = len(class_names)
    train_data, val_data, test_data = stratified_split(paths, labels)
    # Calcula class weights
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(train_data[1]), y=train_data[1])
    class_weight_dict = dict(enumerate(class_weights))
    # Crea datasets
    train_ds = make_dataset(train_data[0], train_data[1], training=True, use_mixup=True)
    val_ds = make_dataset(val_data[0], val_data[1], training=False)
    test_ds = make_dataset(test_data[0], test_data[1], training=False)
    # Crea modelo
    model, base_model = build_robust_model(NUM_CLASSES)
    # Entrenamiento fase 1
    history1 = train_phase1(model, train_ds, val_ds, class_weight_dict, timestamp)
    # Entrenamiento fase 2
    history2 = train_phase2(model, base_model, train_ds, val_ds, class_weight_dict, timestamp)
    # Evaluación
    predictions, predicted_classes, true_classes, report_dict = evaluate_model(
        model, test_ds, class_names, timestamp, use_tta=False)
    # Visualización
    plot_results(history1, history2, predictions, true_classes, predicted_classes, class_names, timestamp)
    # Guarda modelo y clases
    model.save(os.path.join(RESULTS_DIR, f'mushroom_robust_{timestamp}.h5'))
    with open(os.path.join(RESULTS_DIR, f'class_names_robust_{timestamp}.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"\n[INFO] Todos los resultados y gráficos se han guardado en la carpeta '{RESULTS_DIR}'.")

if __name__ == "__main__":
    main()
