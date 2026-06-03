import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
dataset_path = "/Users/adnanaminmir/Desktop/MAIN_PROJECT/Garbage classification"
print("Dataset Exists:", os.path.exists(dataset_path))
print(os.listdir(dataset_path))
import os

ds_file = "/Users/adnanaminmir/Desktop/MAIN_PROJECT/Garbage classification/.DS_Store"

if os.path.exists(ds_file):
    os.remove(ds_file)
    print(".DS_Store removed")
else:
    print("No .DS_Store found")
    
print(os.listdir(dataset_path))    

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)


validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
print(train_generator.class_indices)
print("Number of Classes:", train_generator.num_classes)
print(train_generator.class_indices)

print("Total Classes:", train_generator.num_classes)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)


x = Dropout(0.3)(x)


output = Dense(
    6,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)
model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15
)
plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label="Training Accuracy"
)

plt.plot(
    history.history['val_accuracy'],
    label="Validation Accuracy"
)


plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Smart Waste Classification Accuracy"
)

plt.legend()

plt.show()
model.save(
    "smart_waste_model.h5"
)


print(
    "Model Saved Successfully"
)
img_path = "/Users/adnanaminmir/Desktop/WhatsApp Image 2026-05-31 at 13.50.23.jpeg"


img = image.load_img(
    img_path,
    target_size=(224,224)
)


img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0
prediction = model.predict(
    img_array
)


classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


result = classes[
    np.argmax(prediction)
]


confidence = np.max(
    prediction
) * 100


print(
    "Predicted Waste Category:",
    result
)


print(
    "Confidence:",
    round(confidence,2),
    "%"
)
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Waste Classification Accuracy")

plt.legend()
plt.show()
