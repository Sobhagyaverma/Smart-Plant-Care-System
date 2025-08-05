import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# --- App Configuration ---
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# --- Model and Class Names ---
MODEL_PATH = 'plant_disease_model.h5'
# IMPORTANT: Replace this with the direct download link you created!
MODEL_URL = "https://filebin.net/3v9wy4qf8znv9iiy" 
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# --- Function to download the model ---
def download_file(url, save_path):
    # Check if the file already exists
    if not os.path.exists(save_path):
        with st.spinner('Downloading model... This may take a minute.'):
            try:
                # Make a request to the URL
                r = requests.get(url, stream=True)
                r.raise_for_status() # Raise an exception for bad status codes
                
                # Write the content to the file
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}")
                st.stop()

# --- Load the Model ---
@st.cache_resource
def load_model():
    # Download the model file if it doesn't exist
    download_file(MODEL_URL, MODEL_PATH)
    # Load the downloaded model
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- Prediction Function ---
def predict(image):
    img_array = np.array(image.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0]) * 100
    return predicted_class_name, confidence

# --- Treatment Database (simplified for brevity) ---
treatment_database = {
    "Apple___Apple_scab": {"suggestion": "Apply fungicide.", "prevention": "Prune trees."},
    "Tomato___Late_blight": {"suggestion": "Remove infected plants.", "prevention": "Ensure good airflow."},
    "default": {"suggestion": "No specific suggestion.", "prevention": "Monitor plant health."}
}

# --- App UI ---
st.title("🌿 Plant Disease Detection System")
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Upload Image", "Live Camera"])

if app_mode == "About":
    st.markdown("This application helps in identifying diseases in plants.")
elif app_mode in ["Upload Image", "Live Camera"]:
    image_source = None
    if app_mode == "Upload Image":
        image_source = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        image_source = st.camera_input("Take a picture of the plant leaf")

    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption='Input Image.', use_column_width=True)
        
        with st.spinner('Classifying...'):
            predicted_class, confidence = predict(image)
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.markdown("---")
        st.header("Recommended Actions")
        treatment = treatment_database.get(predicted_class, treatment_database["default"])
        
        with st.expander("🔬 Suggested Treatment"):
            st.write(treatment["suggestion"])
        
        with st.expander("🛡️ Preventive Measures"):
            st.write(treatment["prevention"])

