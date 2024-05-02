import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory where you want to store your dataset
dataset_directory = 'A:/potholeimages'  # Change this to your desired path

# Create the dataset directory if it doesn't exist
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

# Define the model architecture (you may need to customize this)
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification for pothole or not
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess your labeled dataset
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the directory where your images are located
image_directory = os.path.join(dataset_directory, 'images')  # Create an 'images' subdirectory

# Create the 'images' directory if it doesn't exist
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Place your labeled images within their respective class subdirectories
# Example: 'pothole' and 'not_pothole' are class subdirectories
pothole_directory = os.path.join(image_directory, 'pothole')
not_pothole_directory = os.path.join(image_directory, 'not_pothole')

# Create class subdirectories if they don't exist
if not os.path.exists(pothole_directory):
    os.makedirs(pothole_directory)

if not os.path.exists(not_pothole_directory):
    os.makedirs(not_pothole_directory)

# Continue with the rest of your code, including creating and using 'train_generator'
# ...

# Define the number of epochs and batch size for training
epochs = 10
batch_size = 32

# Use 'fit' to train your model with the data generator
model.fit(
    train_datagen.flow_from_directory(
        image_directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'  # For binary classification
    ),
    epochs=epochs
)

# Save the trained model
model.save('pothole_detection_model.h5')
