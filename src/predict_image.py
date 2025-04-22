from keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Cargar modelo entrenado
model_path = Path(__file__).parent.parent / 'trained_model_parameters' / 'final_model.h5'
model = load_model(model_path)
class_labels = ['ESP', 'ARDUINO']  # Cambia si tus clases son otras

# Función para predecir la clase de una imagen
def predict_image(image_path):
    img = Image.open(image_path).resize((300, 180)).convert('RGB')  # Cambié a (300, 180)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalizamos la imagen
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    label = class_labels[top_idx]
    confidence = preds[top_idx] * 100
    return label, confidence

# Función para cargar una nueva imagen
def load_new_image():
    file_path = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg;*.jpeg;*.png")])
    if file_path:
        label, conf = predict_image(file_path)
        pred_label.config(text=f"Predicción: {label} ({conf:.2f}%)")

        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk

# Interfaz gráfica
root = tk.Tk()
root.title("Placas de desarrollo")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

btn = tk.Button(frame, text="📸 Cargar Imagen", command=load_new_image, font=("Arial", 12))
btn.pack(pady=10)

image_label = tk.Label(frame)
image_label.pack()

pred_label = tk.Label(frame, text="", font=("Arial", 14))
pred_label.pack(pady=10)

root.mainloop()
