from keras.models import load_model
import cv2
import numpy as np

# Cargar el modelo desde la ruta correcta
model = load_model("C:\\Users\\aleex\\OneDrive\\Escritorio\\artificial\\final\\trained_model_parameters\\final_model.h5")

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Etiquetas de las clases
class_labels = {0: "ESP32", 1: "Arduino Uno"}  # Mapeo de las clases a los nombres
threshold = 0.6  # Umbral de confianza para la predicción

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()

    if not ret:
        print("No se pudo obtener la imagen de la cámara")
        break

    # Preprocesar la imagen
    img = cv2.resize(frame, (32, 32))  # Cambiar el tamaño a 32x32 (ajustado al modelo entrenado)
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Agregar la dimensión del batch

    # Realizar la predicción
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)  # Obtener la clase predicha
    confidence = np.max(predictions)  # Obtener la confianza de la predicción

    # Si la confianza es baja, indicar que no se detectó nada
    if confidence < threshold:
        label = "Predicción: Nada detectado"
        color = (0, 0, 255)  # Rojo para "Nada detectado"
    else:
        # Obtener la etiqueta correspondiente
        label = f'Predicción: {class_labels[predicted_class[0]]} ({confidence*100:.2f}%)'
        color = (0, 255, 0)  # Verde para predicción confiable

    # Mostrar la clase predicha en el fotograma
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar el fotograma con la predicción
    cv2.imshow("Clasificación en tiempo real", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
