# Proyecto: Reconocimiento de ESP32 vs Arduino Uno

Este proyecto consiste en la creación de una red neuronal convolucional para clasificar imágenes de dos tipos de dispositivos: **ESP32** y **Arduino Uno**. La red fue entrenada utilizando imágenes de ambos dispositivos, y se implementaron técnicas de preprocesamiento de datos para mejorar la precisión del modelo.

## Descripción del Proyecto

La red neuronal fue entrenada para diferenciar entre dos clases:

- **Clase 1**: ESP32
- **Clase 2**: Arduino Uno

El proceso de entrenamiento incluyó las siguientes etapas:

1. **Aumento de Datos (Data Augmentation)**:
   Para mejorar la robustez del modelo y evitar el sobreajuste, se aplicaron técnicas de aumento de datos. Esto incluyó variaciones en la rotación, el zoom y el desplazamiento de las imágenes, creando más datos de entrenamiento a partir de un conjunto limitado.

2. **Redimensionamiento de Imágenes**:
   Después del aumento de datos, todas las imágenes fueron redimensionadas a un tamaño estándar para que pudieran ser procesadas de manera eficiente por la red neuronal.

3. **Uso del Conjunto CIFAR**:
   El modelo fue entrenado utilizando el conjunto de imágenes con el formato y tamaño del conjunto CIFAR, que es ampliamente utilizado para tareas de clasificación de imágenes.

## Proceso de Entrenamiento

El entrenamiento fue realizado utilizando **Keras** y **TensorFlow**, aprovechando la potencia de las redes neuronales convolucionales (CNN). Durante el entrenamiento, se optimizó el modelo para minimizar el error de clasificación y se evaluaron diferentes arquitecturas de red para encontrar la más adecuada para este tipo de imágenes.

