from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from pathlib import Path

def data_augmentation():
    # Rutas de entrada: tus carpetas originales
    input_dirs = {
        'clase1': Path("C:/Users/aleex/OneDrive/Escritorio/artificial/final/Data/clase1"),  # ESP32
        'clase2': Path("C:/Users/aleex/OneDrive/Escritorio/artificial/final/Data/clase2")   # Arduino Uno
    }

    # Ruta de salida para guardar imágenes aumentadas
    output_base = Path("C:/Users/aleex/OneDrive/Escritorio/artificial/final/Data_Augmented")
    output_base.mkdir(exist_ok=True)

    # Generador de aumentaciones
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.4, 1.5]
    )

    # Número de imágenes aumentadas por imagen original
    qty_copies = 10

    for class_name, input_path in input_dirs.items():
        output_dir = output_base / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada imagen de la clase
        for img_file in input_path.glob("*.*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue  # Ignorar archivos que no sean imágenes

            img = load_img(img_file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            prefix = img_file.stem

            i = 0
            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=output_dir,
                save_prefix=prefix + "_aug",
                save_format='jpg'
            ):
                i += 1
                if i >= qty_copies:
                    break

    print("✅ Imágenes aumentadas guardadas en: ", output_base)
