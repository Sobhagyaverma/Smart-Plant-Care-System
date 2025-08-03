import tensorflow as tf
import os

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Correct path to your training data
train_dir = os.path.join('New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)', 'train')

# Load the data from the directory
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Print the class names that were found
class_names = train_dataset.class_names
print("\nFound the following classes:")
print(class_names)
print(f"\nThere are {len(class_names)} classes.")
