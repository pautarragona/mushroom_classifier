"""
Script para limpiar el dataset eliminando imágenes corruptas o problemáticas.
Ejecuta este script si sigues teniendo problemas con imágenes truncadas.
"""

import os
from PIL import Image, ImageFile
import shutil

# Permitir cargar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = 'Mushrooms'
BACKUP_DIR = 'Mushrooms_backup'

def verificar_y_limpiar_imagenes(data_dir, crear_backup=True):
    """
    Verifica todas las imágenes y mueve las corruptas a una carpeta de respaldo.
    """
    if crear_backup and not os.path.exists(BACKUP_DIR):
        print(f"Creando backup del dataset en '{BACKUP_DIR}'...")
        shutil.copytree(data_dir, BACKUP_DIR)
        print(" Backup creado\n")
    
    print("Analizando imágenes del dataset...\n")
    
    corrupted_images = []
    valid_images = 0
    total_images = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                total_images += 1
                filepath = os.path.join(root, file)
                
                try:
                    # Intentar abrir y verificar la imagen
                    with Image.open(filepath) as img:
                        img.verify()
                    
                    # Intentar cargar completamente
                    with Image.open(filepath) as img:
                        img.load()
                        # Verificar que tiene contenido
                        if img.size[0] == 0 or img.size[1] == 0:
                            raise ValueError("Imagen sin dimensiones válidas")
                    
                    valid_images += 1
                    
                except Exception as e:
                    corrupted_images.append((filepath, str(e)))
                    print(f" Corrupta: {os.path.basename(filepath)}")
                    print(f" Error: {str(e)[:80]}")
    
    print(f"\n{'='*70}")
    print(f" RESUMEN:")
    print(f"{'='*70}")
    print(f"Total de imágenes: {total_images}")
    print(f"Imágenes válidas: {valid_images} ({valid_images/total_images*100:.1f}%)")
    print(f"Imágenes corruptas: {len(corrupted_images)} ({len(corrupted_images)/total_images*100:.1f}%)")
    
    if corrupted_images:
        print(f"\n  Se encontraron {len(corrupted_images)} imágenes problemáticas.")
        respuesta = input("\n¿Deseas eliminar las imágenes corruptas? (s/n): ").lower()
        
        if respuesta == 's':
            print("\n  Eliminando imágenes corruptas...")
            for img_path, error in corrupted_images:
                try:
                    os.remove(img_path)
                    print(f"    Eliminada: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"    No se pudo eliminar {os.path.basename(img_path)}: {e}")
            
            print(f"\n Se eliminaron {len(corrupted_images)} imágenes corruptas.")
            print(f"  (Puedes recuperarlas desde '{BACKUP_DIR}' si lo necesitas)")
        else:
            print("\n No se eliminó ninguna imagen.")
    else:
        print("\n ¡Todas las imágenes están en buen estado!")
    
    print(f"{'='*70}\n")

def intentar_reparar_imagenes(data_dir):
    """
    Intenta reparar imágenes truncadas convirtiéndolas a un formato nuevo.
    """
    print(" Intentando reparar imágenes truncadas...\n")
    
    reparadas = 0
    no_reparables = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                filepath = os.path.join(root, file)
                
                try:
                    # Intentar cargar con el modo que permite truncado
                    with Image.open(filepath) as img:
                        img.load()
                except:
                    try:
                        # Intentar reparar guardando de nuevo
                        print(f" Reparando: {os.path.basename(filepath)}")
                        with Image.open(filepath) as img:
                            # Convertir a RGB si es necesario
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Guardar con un nombre temporal
                            temp_path = filepath + '.temp.jpg'
                            img.save(temp_path, 'JPEG', quality=95)
                        
                        # Reemplazar la imagen original
                        os.remove(filepath)
                        os.rename(temp_path, filepath)
                        reparadas += 1
                        print(f"   Reparada exitosamente")
                        
                    except Exception as e:
                        no_reparables.append((filepath, str(e)))
                        print(f"   No se pudo reparar: {str(e)[:50]}")
    
    print(f"\n{'='*70}")
    print(f"Imágenes reparadas: {reparadas}")
    print(f"Imágenes no reparables: {len(no_reparables)}")
    print(f"{'='*70}\n")
    
    return reparadas, no_reparables

def main():
    """
    Función principal del script de limpieza.
    """
    print("\n" + "="*70)
    print(" "*15 + "LIMPIEZA DE DATASET DE SETAS")
    print("="*70 + "\n")
    
    if not os.path.exists(DATA_DIR):
        print(f" Error: No se encuentra el directorio '{DATA_DIR}'")
        return
    
    print("Opciones disponibles:")
    print("1. Verificar y eliminar imágenes corruptas")
    print("2. Intentar reparar imágenes truncadas")
    print("3. Verificar solamente (sin cambios)")
    print()
    
    opcion = input("Selecciona una opción (1/2/3): ").strip()
    
    if opcion == '1':
        verificar_y_limpiar_imagenes(DATA_DIR, crear_backup=True)
    elif opcion == '2':
        # Crear backup primero
        if not os.path.exists(BACKUP_DIR):
            print(f" Creando backup en '{BACKUP_DIR}'...")
            shutil.copytree(DATA_DIR, BACKUP_DIR)
            print(" Backup creado\n")
        
        reparadas, no_reparables = intentar_reparar_imagenes(DATA_DIR)
        
        if no_reparables:
            print("\n¿Deseas eliminar las imágenes no reparables? (s/n): ", end='')
            if input().lower() == 's':
                for img_path, _ in no_reparables:
                    try:
                        os.remove(img_path)
                        print(f" Eliminada: {os.path.basename(img_path)}")
                    except:
                        pass
    elif opcion == '3':
        verificar_y_limpiar_imagenes(DATA_DIR, crear_backup=False)
    else:
        print(" Opción no válida")

if __name__ == "__main__":
    main()
