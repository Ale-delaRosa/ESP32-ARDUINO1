from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def cifar_classification(data_path):

    """
    Esta función entrena una red neuronal convolucional (CNN) para clasificar imágenes
    en dos clases: ESP32 (clase1) y Arduino Uno (clase2).
    """
    current_dir = Path(__file__).parent
    models_dir = current_dir.parent / 'trained_model_parameters'

    train_dir = Path(r'C:\\Users\\aleex\\OneDrive\\Escritorio\\artificial\\final\\Data_Resized\\train')
    valid_dir = Path(r'C:\\Users\\aleex\\OneDrive\\Escritorio\\artificial\\final\Data_Resized\\valid')
    test_dir  = Path(r'C:\\Users\\aleex\\OneDrive\\Escritorio\\artificial\\final\\Data_Resized\\test')

    # Aumentación y normalización
    data_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.4, 1.5]
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    # Cargar datasets desde carpetas
    target_size = (300, 180)  # Cambié esto a 300x180 para tus imágenes
    batch_size = 32

    train_data = data_generator.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_data = test_gen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    n_clases = train_data.num_classes
    filtros = 32
    regularizers_w = 1e-4

    # Modelo
    model = Sequential()
    model.add(Conv2D(filtros, (3, 3), strides=(1, 1), padding='same',
                     kernel_regularizer=regularizers.l2(regularizers_w),
                     input_shape=(180, 300, 3)))  # Aquí cambiamos a (180, 300)
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filtros, (3, 3), padding='same', kernel_regularizer=regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filtros * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(n_clases, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='accuracy', patience=2, verbose=1)

    dt = datetime.now()
    ts = datetime.timestamp(dt)
    model_checkpoint = ModelCheckpoint(
        filepath=str(models_dir / f'bestcifar10{ts}.h5'),  # Cambiado a .h5
        monitor='accuracy',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_data,
        epochs=80,
        validation_data=valid_data,
        callbacks=[model_checkpoint],
        verbose=2
    )

    # Guardar el modelo final
    model.save(models_dir / 'final_model.h5')

    # Gráficas
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Precisión del modelo')
    plt.show()

    # Evaluar
    score = model.evaluate(test_data, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
