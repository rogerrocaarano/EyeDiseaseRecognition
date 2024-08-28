import pathlib
import tensorflow as tf
from keras import Sequential
from keras.applications.densenet import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import datetime

DATASET_FIT_PATH = pathlib.Path('dataset/fit').resolve()
DATASET_TEST_PATH = pathlib.Path('dataset/test').resolve()

MODEL_OUTPUT_PATH = pathlib.Path('models').resolve()

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_FIT_PATH,
    color_mode='grayscale',
    batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_TEST_PATH,
    color_mode='grayscale',
    batch_size=32)

class_names = train_ds.class_names
print(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(256, 256, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=['accuracy']
)

lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.1,
    min_lr=1e-5
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=[lr_reduction]
)

# obtener fecha y hora actual

now = datetime.datetime.now()
# guardar modelo
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
time = now.strftime('%Y-%m-%d_%H-%M-%S')
model.save(MODEL_OUTPUT_PATH / f"model_{time}")

history_file = MODEL_OUTPUT_PATH / f"model_{time}" / f"history.txt"
with open(history_file, "w") as f:
    f.write(f"Modelo entrenado en {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Clases: {class_names}\n")
    f.write(f"Resultados del entrenamiento:\n")
    f.write(str(history.history))

classes_file = MODEL_OUTPUT_PATH / f"model_{time}" / f"classes.txt"
with open(classes_file, "w") as f:
    f.write(str(class_names))

print(f"Modelo guardado en: {MODEL_OUTPUT_PATH}")
