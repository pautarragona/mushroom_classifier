import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Cambia esto a la ruta de tu carpeta principal de setas
DATASET_PATH = "./Mushrooms" 
IMG_SIZE = 64       # 64x64 es un buen balance para estabilidad. 128 es posible pero más difícil de entrenar.
BATCH_SIZE = 32
NOISE_DIM = 128     # Dimensión del vector de ruido latente
EPOCHS = 50         # Aumentar para mejores resultados (ej. 200-500)

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
print(f"Cargando imágenes desde {DATASET_PATH}...")

# Usamos image_dataset_from_directory que infiere etiquetas de las subcarpetas
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    label_mode='int',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = dataset.class_names
num_classes = len(class_names)
print(f"Clases detectadas ({num_classes}): {class_names}")

# Normalizar imágenes a [-1, 1] para la GAN
normalization_layer = layers.Rescaling(scale=1./127.5, offset=-1)
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

# ==========================================
# 2. DEFINICIÓN DE LA cGAN
# ==========================================

def build_generator(latent_dim, num_classes):
    # Entrada de etiqueta (label)
    label_input = layers.Input(shape=(1,), name='label_input')
    # Embedding: transforma la etiqueta (int) a un vector denso
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Dense(4 * 4 * 1)(label_embedding) # Proyectar a 4x4
    label_embedding = layers.Reshape((4, 4, 1))(label_embedding)

    # Entrada de ruido (noise)
    noise_input = layers.Input(shape=(latent_dim,), name='noise_input')
    noise_gen = layers.Dense(4 * 4 * 256)(noise_input)
    noise_gen = layers.Reshape((4, 4, 256))(noise_gen)

    # Concatenar ruido y etiqueta
    merge = layers.Concatenate()([noise_gen, label_embedding])

    # Upsampling (Aumentar resolución progresivamente)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge) # 8x8
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x) # 16x16
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x) # 32x32
    x = layers.LeakyReLU(0.2)(x)
    
    # Capa de salida (64x64x3)
    out_layer = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x) # 64x64

    return models.Model([noise_input, label_input], out_layer, name="Generator")

def build_discriminator(img_size, num_classes):
    # Entrada de imagen
    img_input = layers.Input(shape=(img_size, img_size, 3), name='img_input')
    
    # Entrada de etiqueta
    label_input = layers.Input(shape=(1,), name='label_input')
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Dense(img_size * img_size * 1)(label_embedding)
    label_embedding = layers.Reshape((img_size, img_size, 1))(label_embedding)
    
    # Concatenar imagen y etiqueta (La etiqueta actúa como un canal extra)
    merge = layers.Concatenate()([img_input, label_embedding])
    
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(merge)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Flatten()(x)
    out_layer = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model([img_input, label_input], out_layer, name="Discriminator")

# Construir modelos
generator = build_generator(NOISE_DIM, num_classes)
discriminator = build_discriminator(IMG_SIZE, num_classes)

# Optimizadores
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO (Custom Loop)
# ==========================================

@tf.function
def train_step(real_images, labels):
    batch_size = tf.shape(real_images)[0]
    
    # 1. Entrenar Discriminador
    random_latent_vectors = tf.random.normal(shape=(batch_size, NOISE_DIM))
    generated_images = generator([random_latent_vectors, labels], training=True)
    
    # Etiquetas para el discriminador (1=real, 0=fake)
    # Se añade un poco de ruido a las etiquetas reales para estabilizar (smoothing)
    labels_real = tf.ones((batch_size, 1)) * 0.9 
    labels_fake = tf.zeros((batch_size, 1))
    
    with tf.GradientTape() as tape:
        predictions_real = discriminator([real_images, labels], training=True)
        predictions_fake = discriminator([generated_images, labels], training=True)
        
        d_loss_real = loss_fn(labels_real, predictions_real)
        d_loss_fake = loss_fn(labels_fake, predictions_fake)
        d_loss = d_loss_real + d_loss_fake
        
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    
    # 2. Entrenar Generador
    with tf.GradientTape() as tape:
        random_latent_vectors = tf.random.normal(shape=(batch_size, NOISE_DIM))
        generated_images = generator([random_latent_vectors, labels], training=True)
        
        # El generador quiere engañar al discriminador (quiere que diga 1)
        predictions_fake = discriminator([generated_images, labels], training=True)
        g_loss = loss_fn(tf.ones((batch_size, 1)), predictions_fake)
        
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    
    return d_loss, g_loss

print("Iniciando entrenamiento...")
for epoch in range(EPOCHS):
    d_loss_avg = 0
    g_loss_avg = 0
    batches = 0
    
    for images, labels in dataset:
        d_loss, g_loss = train_step(images, labels)
        d_loss_avg += d_loss
        g_loss_avg += g_loss
        batches += 1
        
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss_avg/batches:.4f} | G Loss: {g_loss_avg/batches:.4f}")

# ==========================================
# 4. GENERACIÓN Y GUARDADO
# ==========================================
OUTPUT_FOLDER = "mushrooms_augmented"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMAGES_PER_CLASS = 50 # Cuántas imágenes nuevas quieres por carpeta

print(f"Generando {IMAGES_PER_CLASS} imágenes nuevas por clase...")

for class_idx, name in enumerate(class_names):
    class_folder = os.path.join(OUTPUT_FOLDER, name)
    os.makedirs(class_folder, exist_ok=True)
    
    # Generar batch de ruido y etiquetas repetidas
    noise = tf.random.normal(shape=(IMAGES_PER_CLASS, NOISE_DIM))
    labels = tf.constant([class_idx] * IMAGES_PER_CLASS, dtype=tf.int32)
    
    generated_imgs = generator([noise, labels], training=False)
    
    # Des-normalizar de [-1, 1] a [0, 255]
    generated_imgs = (generated_imgs * 127.5) + 127.5
    generated_imgs = generated_imgs.numpy().astype(np.uint8)
    
    for i, img in enumerate(generated_imgs):
        save_path = os.path.join(class_folder, f"aug_{i}.png")
        tf.keras.utils.save_img(save_path, img)

print(f"¡Hecho! Imágenes guardadas en la carpeta '{OUTPUT_FOLDER}'.")
