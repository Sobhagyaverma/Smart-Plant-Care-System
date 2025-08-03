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

print("\n✅ Data loading and preparation complete.")