from PIL import Image
import os
from pathlib import Path

# Carpeta principal que contiene las dos carpetas de clases (ESP32 y Arduino)
input_folder = Path("C:\\Users\\aleex\\OneDrive\\Escritorio\\artificial\\final\\Data_Augmented")
# Carpeta principal donde se guardarán las imágenes redimensionadas
output_folder = Path("C:\\Users\\aleex\\OneDrive\\Escritorio\\artificial\\final\\Data_Resized")

# Si la carpeta de salida no existe, créala
output_folder.mkdir(parents=True, exist_ok=True)

# Recorre las subcarpetas (cada una es una clase)
for class_folder in input_folder.iterdir():
    if class_folder.is_dir():
        # Crear una subcarpeta en la carpeta de salida para cada clase
        class_output_folder = output_folder / class_folder.name
        class_output_folder.mkdir(parents=True, exist_ok=True)

        # Redimensionar todas las imágenes dentro de cada subcarpeta
        for img_file in class_folder.glob("*.*"):  # Recorre todas las imágenes
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Abrir imagen
                img = Image.open(img_file)
                
                # Redimensionar a 300x300
                img_resized = img.resize((180, 300))
                
                # Guardar la imagen redimensionada en la subcarpeta correspondiente
                output_path = class_output_folder / img_file.name
                img_resized.save(output_path)

print("Redimensionado completado.")
