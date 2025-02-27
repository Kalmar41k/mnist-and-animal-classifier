"""
This script downloads the Animals-10 dataset from Kaggle, processes the data by renaming class folders,
balances the dataset by limiting the number of images per class, augments underrepresented classes,
splits the dataset into training and validation sets, and trains a fine-tuned VGG16 model for image classification.

Main steps:
1. Download and preprocess the dataset (rename classes, limit image count per class, augment images if necessary).
2. Split the dataset into training and validation sets.
3. Create image generators for training and validation.
4. Load and fine-tune a pre-trained VGG16 model.
5. Train the model with early stopping and learning rate reduction.
6. Evaluate the model on the validation set.
7. Save the trained model and class labels.
"""

import os
import kaggle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Download and unzip the dataset from Kaggle
kaggle.api.dataset_download_files('alessiocorrado99/animals10', path='data/animals_dataset', unzip=True)

# Mapping Italian class names to English
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider"
}

DATASET_PATH = "data/animals_dataset/raw-img"

# Rename class directories to English names
for old_name, new_name in translate.items():
    old_path = os.path.join(DATASET_PATH, old_name)
    new_path = os.path.join(DATASET_PATH, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)

classes = os.listdir(DATASET_PATH)

print("Translated classes:")
print(classes)

# Reduce the number of images per class to a target size
oversampled_classes = []
target_size = 2500

for cls in os.listdir(DATASET_PATH):
    class_size = len(os.listdir(os.path.join(DATASET_PATH, cls)))
    if class_size > target_size:
        oversampled_classes.append(cls)

for cls in oversampled_classes:
    class_path = os.path.join(DATASET_PATH, cls)
    images = os.listdir(class_path)
    
    if len(images) > target_size:
        images_to_remove = random.sample(images, len(images) - target_size)
        for img in images_to_remove:
            os.remove(os.path.join(class_path, img))

    print(f"{cls}: redused to {target_size} images")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_class(class_name, target_size):
    class_path = os.path.join(DATASET_PATH, class_name)
    images = os.listdir(class_path)

    while len(images) < target_size:
        img_path = os.path.join(class_path, random.choice(images))
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))  
        img_array = np.array(img)

        if img_array.shape[-1] != 3:
            continue  

        img_array = img_array.reshape((1, 224, 224, 3))

        augmented_img = next(datagen.flow(img_array, batch_size=1))[0].astype(np.uint8)
        new_img = Image.fromarray(augmented_img)

        new_img_name = f"aug_{len(images)}.jpg"
        new_img_path = os.path.join(class_path, new_img_name)

        new_img.save(new_img_path)
        images.append(new_img_name)

    print(f"{class_name}: increased to {target_size} images")

# Augment underrepresented classes
for cls in os.listdir(DATASET_PATH):
    if cls not in oversampled_classes:
        class_size = len(os.listdir(os.path.join(DATASET_PATH, cls)))
        if class_size < target_size:
            augment_class(cls, target_size=target_size)

# Splitting dataset into training and validation sets
TRAIN_PATH = "data/animals_dataset/train"
VAL_PATH = "data/animals_dataset/val"

def create_dir_structure(base_path, classes):
    for class_name in classes:
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

create_dir_structure(TRAIN_PATH, classes)
create_dir_structure(VAL_PATH, classes)

for class_name in classes:
    class_folder = os.path.join(DATASET_PATH, class_name)
    images = os.listdir(class_folder)
    
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for image in train_images:
        shutil.move(os.path.join(class_folder, image), os.path.join(TRAIN_PATH, class_name, image))

    for image in val_images:
        shutil.move(os.path.join(class_folder, image), os.path.join(VAL_PATH, class_name, image))

print("The dataset has been successfully divided into train and val.")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(TRAIN_PATH),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(VAL_PATH),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb'
)

# Model setup using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks = [early_stopping, reduce_lr]
)

# Fine-tuning the model
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine_tuning = model.fit(
    train_generator,
    epochs=5,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Model evaluation
test_generator = val_datagen.flow_from_directory(
    os.path.join(VAL_PATH),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle = False
)

y_true = test_generator.classes

y_pred_probs = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(test_generator.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))

# Save the trained model
model.save("models/vgg16_finetuned.h5")

# Save the class labels
labels_path = 'data/class_labels.txt'

with open(labels_path, 'w') as f:
    for label in class_labels:
        f.write(f"{label}\n")