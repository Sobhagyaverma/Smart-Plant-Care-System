import tensorflow as tf
import os

# --- Constants ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = 'plant_disease_model.h5'
VALID_DIR = os.path.join('New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)', 'valid')

# --- Main Execution ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Loading validation data from: {VALID_DIR}")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  # No need to shuffle for evaluation
)

print("\nEvaluating model on the validation dataset...")
loss, accuracy = model.evaluate(validation_dataset)

print("\n✅ Evaluation Complete.")
print(f"   Validation Loss: {loss:.4f}")
print(f"   Validation Accuracy: {accuracy * 100:.2f}%")