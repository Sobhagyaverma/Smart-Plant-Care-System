import tensorflow as tf
import os

# 1. Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5 # Start with a small number to test

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

# 5. Get class names before optimizing
class_names = train_dataset.class_names
print(f"\nFound {len(class_names)} classes.")
print(class_names)

# 6. Add performance optimizations
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("\n✅ Data loading and preparation complete.")

# 7. Build the model architecture
print("\nBuilding model architecture...")
num_classes = len(class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
  
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 8. Compile the model
print("\nCompiling model...")
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
print("✅ Model compiled successfully.")
model.summary()

# 9. Train the model
print("\nStarting model training...")
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=EPOCHS
)

# 10. Save the trained model
print("\n✅ Training complete.")
model.save('plant_disease_model.h5')
print("✅ Model saved to plant_disease_model.h5")