import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- App Configuration ---
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# --- Treatment Database ---
# A simple dictionary to store treatment suggestions for each disease
treatment_database = {
    "Apple___Apple_scab": {
        "suggestion": "Remove and destroy infected leaves and fruit. Apply a fungicide containing myclobutanil or captan.",
        "prevention": "Ensure good air circulation by pruning trees. Water at the base to avoid wet leaves. Rake up and dispose of fallen leaves in autumn."
    },
    "Apple___Black_rot": {
        "suggestion": "Prune out dead or cankered branches. Remove and dispose of mummified fruit. Apply a fungicide during the growing season.",
        "prevention": "Maintain tree health with proper watering and fertilization. Avoid wounding the tree."
    },
    "Tomato___Bacterial_spot": {
        "suggestion": "Apply copper-based bactericides. Remove heavily infected plants to prevent spread.",
        "prevention": "Use disease-free seeds and transplants. Rotate crops, avoiding planting tomatoes or peppers in the same spot for at least a year. Avoid overhead watering."
    },
    "Tomato___Late_blight": {
        "suggestion": "Immediately remove and destroy infected plants. Apply fungicides containing chlorothalonil, mancozeb, or copper.",
        "prevention": "Ensure good airflow. Water early in the day at the soil level. Monitor weather forecasts for conditions favorable to blight."
    },
    "Corn_(maize)___healthy": {
        "suggestion": "Your plant appears to be healthy.",
        "prevention": "Continue with good watering practices, ensure adequate sunlight, and monitor regularly for any signs of pests or disease."
    },
    "default": {
        "suggestion": "No specific treatment suggestion available for this condition.",
        "prevention": "General best practices include ensuring proper watering, adequate sunlight, good soil drainage, and regular monitoring."
    }
}


# --- Load the Model and Class Names ---
@st.cache_resource
def load_model_and_classes():
    """
    Loads the trained model and class names once and caches them.
    """
    model = tf.keras.models.load_model('plant_disease_model.h5')
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    return model, class_names

model, class_names = load_model_and_classes()

# --- Prediction Function ---
def predict(image):
    """
    Takes an image, preprocesses it, and returns the predicted class and confidence.
    """
    img_array = np.array(image.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100
    return predicted_class_name, confidence

# --- App UI ---
st.title("🌿 Plant Disease Detection System")
st.write("Choose an option below to check for plant diseases.")

st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["About", "Upload Image", "Live Camera"])

if app_mode == "About":
    st.sidebar.success('To begin, select an option from the dropdown menu.')
    st.markdown("This application helps in identifying diseases in plants using a deep learning model.")
    st.markdown("You can either upload an image of a plant leaf or use your device's camera for a live prediction.")

elif app_mode == "Upload Image" or app_mode == "Live Camera":
    if app_mode == "Upload Image":
        st.sidebar.success('You have selected the Upload Image mode.')
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        image_source = uploaded_file
    else:
        st.sidebar.info('You have selected the Live Camera mode. Please grant camera access when prompted.')
        img_file_buffer = st.camera_input("Take a picture of the plant leaf")
        image_source = img_file_buffer

    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption='Input Image.', use_column_width=True)
        
        with st.spinner('Classifying...'):
            predicted_class, confidence = predict(image)
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Display treatment suggestions
        st.markdown("---")
        st.header("Recommended Actions")
        treatment = treatment_database.get(predicted_class, treatment_database["default"])
        
        with st.expander("🔬 Suggested Treatment"):
            st.write(treatment["suggestion"])
        
        with st.expander("🛡️ Preventive Measures"):
            st.write(treatment["prevention"])
