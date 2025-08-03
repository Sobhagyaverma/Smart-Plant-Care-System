import tensorflow as tf
import os

# 1. Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. Define paths
base_dir = os.path.join('New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# 3. Load training data
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 4. Load validation data
print("Loading validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 5. GET CLASS NAMES **BEFORE** OPTIMIZING
class_names = train_dataset.class_names
print(f"\nFound {len(class_names)} classes.")
print(class_names)

# 6. Add performance optimizations
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# (Keep all the code from before)

# ... previous code ends here
print("\n✅ Data loading and preparation complete.")


# 7. Build the model architecture
print("\nBuilding model architecture...")
num_classes = len(class_names)

model = tf.keras.Sequential([
  # Input layer: Rescale pixel values from [0, 255] to [0, 1]
  tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
  
  # First Convolutional Block
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  # Second Convolutional Block
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  # Third Convolutional Block
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  # Flatten the results to feed into a dense layer
  tf.keras.layers.Flatten(),
  
  # Dense layer for classification
  tf.keras.layers.Dense(512, activation='relu'),
  
  # Output layer: Must have the same number of units as classes
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 8. Print the model summary
print("\n✅ Model architecture built.")
model.summary()