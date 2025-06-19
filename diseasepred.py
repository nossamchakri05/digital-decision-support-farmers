import os
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set paths to dataset
train_images_dir = 'Data/train_images'
train_csv_path = 'Data/train.csv'
label_map_path = 'Data/label_num_to_disease_map.json'

# Load the label mapping
with open(label_map_path, 'r') as f:
    label_map = json.load(f)

# Load the CSV containing the image filenames and their corresponding labels
df = pd.read_csv(train_csv_path)

# Map disease labels from string to integers using the label map
label_dict = {v: k for k, v in label_map.items()}
df['label'] = df['label'].map(label_dict)

# Image preprocessing function to load and normalize images
def preprocess_image(img_path, target_size=(224, 224)):
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found.")
        return None  # Return None if file does not exist
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    return img_array

# Create a list of image paths and corresponding labels
image_paths = [os.path.join(train_images_dir, fname) for fname in df['image_id']]
labels = df['label'].values

# Preprocess all images, skipping missing files
images = []
valid_labels = []  # To store the labels corresponding to the valid images
for img_path, label in zip(image_paths, labels):
    img_array = preprocess_image(img_path)
    if img_array is not None:
        images.append(img_array)
        valid_labels.append(label)

# Check if any images were processed, otherwise raise an error
if not images:
    raise ValueError("No valid images found. Please check the dataset.")

# Convert list to numpy array
images = np.array(images)
valid_labels = np.array(valid_labels)

# Check for any invalid labels
if np.any(np.isnan(valid_labels)) or np.any(valid_labels < 0) or np.any(valid_labels >= len(label_map)):
    print("Warning: Invalid labels detected.")
    valid_labels = np.clip(valid_labels, 0, len(label_map)-1)  # Ensure labels are within bounds

# One-hot encode labels
labels = to_categorical(valid_labels, num_classes=len(label_map))

# Split the dataset into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation (for training data only)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on training data
datagen.fit(X_train)

# Load the pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers for our classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout for regularization
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(label_map), activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of ResNet50 (only train the custom layers)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define a checkpoint to save the best model based on validation loss
checkpoint = ModelCheckpoint('cassava_disease_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    steps_per_epoch=len(X_train) // 32,
    validation_steps=len(X_val) // 32,
    callbacks=[checkpoint]
)

# Optionally, save the final model after training
model.save('cassava_disease_model_final.h5')

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
print(f'Validation Accuracy: {val_acc * 100:.2f}%')
