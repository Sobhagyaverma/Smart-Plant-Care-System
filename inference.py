import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

# Load models
cnn = tf.keras.models.load_model("cnn_mobilenetv2.keras")
svm = joblib.load("svm_linear.pkl")
scaler = joblib.load("scaler.pkl")

print("✅ Models loaded")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Feature extractor
feature_extractor = Model(
    inputs=cnn.input,
    outputs=cnn.layers[-3].output
)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = feature_extractor.predict(img, verbose=0)
    features = scaler.transform(features)

    pred = svm.predict(features)[0]
    return labels[pred]

# Test
print("Prediction:", predict_image("AppleCedarRust1.JPG"))