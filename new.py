import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import random
import os
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="Agro Guard: Smart Crop Care System", layout="wide")

# --- Treatment Database (Expanded for all classes) ---
treatment_database = {
    # Apple Diseases
    "Apple___Apple_scab": {
        "suggestion": "Remove and destroy infected leaves and fruit. Apply a fungicide containing myclobutanil or captan in early spring.",
        "prevention": "Ensure good air circulation by pruning trees. Water at the base to avoid wet leaves. Rake up and dispose of fallen leaves in autumn."
    },
    "Apple___Black_rot": {
        "suggestion": "Prune out dead or cankered branches well below the infected area. Remove and dispose of mummified fruit.",
        "prevention": "Maintain tree health with proper watering and fertilization. Avoid wounding the tree. Apply protective fungicides from bud break until petal fall."
    },
    "Apple___Cedar_apple_rust": {
        "suggestion": "Apply a fungicide (e.g., myclobutanil) starting at bloom and continuing every 7-10 days. Prune out visible galls.",
        "prevention": "Remove nearby cedar trees if possible, as they are the alternate host. Plant rust-resistant apple varieties."
    },
    "Apple___healthy": {
        "suggestion": "Your apple tree appears to be healthy.",
        "prevention": "Continue with good watering practices, ensure adequate sunlight, and monitor regularly for any signs of pests or disease."
    },
    # Blueberry
    "Blueberry___healthy": {
        "suggestion": "Your blueberry plant appears to be healthy.",
        "prevention": "Maintain acidic soil (pH 4.5-5.5), ensure good drainage, and provide consistent moisture. Mulch to retain moisture and control weeds."
    },
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": {
        "suggestion": "Apply a fungicide containing sulfur, potassium bicarbonate, or neem oil at the first sign of disease.",
        "prevention": "Prune for good air circulation. Plant in a sunny location. Avoid excessive nitrogen fertilizer."
    },
    "Cherry_(including_sour)___healthy": {
        "suggestion": "Your cherry tree appears to be healthy.",
        "prevention": "Ensure well-drained soil and good air circulation. Protect from birds with netting as fruit ripens."
    },
    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "suggestion": "Apply a foliar fungicide when the disease first appears, especially on susceptible hybrids.",
        "prevention": "Rotate crops with non-grass species. Till the soil to bury crop residue. Plant resistant corn hybrids."
    },
    "Corn_(maize)___Common_rust_": {
        "suggestion": "Fungicide application is usually not necessary for common rust unless it appears early on susceptible hybrids.",
        "prevention": "Plant resistant corn varieties. Most field corn hybrids have good resistance."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "suggestion": "Apply a foliar fungicide if the disease is present on the third leaf below the ear leaf or higher on 50% of the plants.",
        "prevention": "Choose resistant hybrids. Practice crop rotation and tillage to reduce residue."
    },
    "Corn_(maize)___healthy": {
        "suggestion": "Your corn plant appears to be healthy.",
        "prevention": "Ensure consistent watering, especially during tasseling and silking. Provide adequate nitrogen fertilizer."
    },
    # Grape
    "Grape___Black_rot": {
        "suggestion": "Apply a fungicide (e.g., mancozeb, captan) at regular intervals, starting when new shoots are 2-4 inches long.",
        "prevention": "Remove and destroy all mummified berries and infected canes during dormancy. Improve air circulation through pruning."
    },
    "Grape___Esca_(Black_Measles)": {
        "suggestion": "There is no cure for Esca. Prune out and destroy infected parts of the vine well below the symptomatic area.",
        "prevention": "Protect pruning wounds with a sealant. Avoid pruning during wet weather. Maintain vine health."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "suggestion": "Apply a broad-spectrum fungicide. This disease is often minor and may not require treatment unless severe.",
        "prevention": "Ensure good air circulation through proper pruning and canopy management. Rake and destroy fallen leaves."
    },
    "Grape___healthy": {
        "suggestion": "Your grapevine appears to be healthy.",
        "prevention": "Practice good pruning techniques to manage the canopy. Ensure well-drained soil and adequate sunlight."
    },
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "suggestion": "There is no cure for Citrus Greening. Remove and destroy the infected tree immediately to prevent spread to other trees.",
        "prevention": "Control the Asian citrus psyllid, the insect that spreads the disease, using insecticides. Plant certified disease-free trees."
    },
    # Peach
    "Peach___Bacterial_spot": {
        "suggestion": "Apply bactericides containing copper during the dormant season. Prune to improve air circulation.",
        "prevention": "Plant resistant varieties. Avoid excessive nitrogen fertilization. Maintain tree vigor."
    },
    "Peach___healthy": {
        "suggestion": "Your peach tree appears to be healthy.",
        "prevention": "Follow a regular spray schedule for common pests and diseases. Prune annually to maintain an open center for good light and air flow."
    },
    # Pepper & Bell
    "Pepper,_bell___Bacterial_spot": {
        "suggestion": "Apply copper-based sprays. Remove infected leaves and plants to reduce spread.",
        "prevention": "Use clean, certified seed. Rotate crops. Avoid working with plants when they are wet."
    },
    "Pepper,_bell___healthy": {
        "suggestion": "Your pepper plant appears to be healthy.",
        "prevention": "Provide consistent moisture and well-drained soil. Support plants with stakes or cages to prevent branches from breaking."
    },
    # Potato
    "Potato___Early_blight": {
        "suggestion": "Apply a fungicide containing chlorothalonil or mancozeb at the first sign of disease.",
        "prevention": "Rotate crops. Use certified disease-free seed potatoes. Ensure good nutrition and water to keep plants vigorous."
    },
    "Potato___Late_blight": {
        "suggestion": "Immediately remove and destroy infected plants. Apply fungicides proactively, especially during cool, wet weather.",
        "prevention": "Plant certified disease-free seed potatoes. Ensure good air circulation and avoid overhead watering."
    },
    "Potato___healthy": {
        "suggestion": "Your potato plant appears to be healthy.",
        "prevention": "Practice hilling (piling soil up around the base) to protect tubers from light and pests. Monitor for Colorado potato beetles."
    },
    # Raspberry
    "Raspberry___healthy": {
        "suggestion": "Your raspberry plant appears to be healthy.",
        "prevention": "Prune canes annually after they have finished fruiting. Ensure good air circulation and weed control."
    },
    # Soybean
    "Soybean___healthy": {
        "suggestion": "Your soybean plant appears to be healthy.",
        "prevention": "Practice crop rotation. Ensure good soil drainage. Plant at the recommended time for your region."
    },
    # Squash
    "Squash___Powdery_mildew": {
        "suggestion": "Apply fungicides like sulfur, neem oil, or potassium bicarbonate at the first sign of the disease.",
        "prevention": "Plant resistant varieties. Ensure good air circulation and sunlight exposure. Water the soil, not the leaves."
    },
    # Strawberry
    "Strawberry___Leaf_scorch": {
        "suggestion": "Remove and destroy infected leaves. Apply a fungicide if the disease is severe.",
        "prevention": "Ensure good air circulation and sunlight. Plant in well-drained soil. Renovate strawberry beds after harvest."
    },
    "Strawberry___healthy": {
        "suggestion": "Your strawberry plant appears to be healthy.",
        "prevention": "Use mulch to keep fruit off the ground and conserve moisture. Protect blossoms from late frosts."
    },
    # Tomato
    "Tomato___Bacterial_spot": {
        "suggestion": "Apply copper-based bactericides. Remove heavily infected plants to prevent spread.",
        "prevention": "Use disease-free seeds and transplants. Rotate crops. Avoid overhead watering."
    },
    "Tomato___Early_blight": {
        "suggestion": "Prune off the lower leaves. Apply fungicides containing chlorothalonil or mancozeb.",
        "prevention": "Mulch at the base of plants. Stake or cage plants to improve air circulation. Rotate crops."
    },
    "Tomato___Late_blight": {
        "suggestion": "Immediately remove and destroy infected plants. Apply fungicides containing chlorothalonil or copper.",
        "prevention": "Ensure good airflow. Water early in the day at the soil level. Monitor weather forecasts."
    },
    "Tomato___Leaf_Mold": {
        "suggestion": "Improve air circulation by pruning and spacing plants. Apply a fungicide if necessary.",
        "prevention": "Water at the base of plants. Provide good ventilation, especially in greenhouses. Use resistant varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "suggestion": "Remove and destroy infected lower leaves. Apply fungicides containing chlorothalonil.",
        "prevention": "Rotate crops. Mulch around the base of plants. Water the soil, not the leaves."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "suggestion": "Spray plants with a strong stream of water to dislodge mites. Apply insecticidal soap or neem oil.",
        "prevention": "Keep plants well-watered to reduce stress. Encourage beneficial insects like ladybugs."
    },
    "Tomato___Target_Spot": {
        "suggestion": "Apply fungicides containing chlorothalonil or mancozeb. Prune to improve air circulation.",
        "prevention": "Rotate crops. Remove crop debris after harvest. Ensure good air circulation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "suggestion": "There is no cure. Remove and destroy infected plants immediately to prevent spread.",
        "prevention": "Control whiteflies, the insects that transmit the virus. Use reflective mulch. Plant virus-resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "suggestion": "There is no cure. Remove and destroy infected plants immediately.",
        "prevention": "Wash hands thoroughly before handling plants. Do not use tobacco products near tomato plants. Use virus-resistant varieties."
    },
    "Tomato___healthy": {
        "suggestion": "Your tomato plant appears to be healthy.",
        "prevention": "Provide consistent watering. Stake or cage plants for support. Fertilize regularly."
    },
    # Default fallback
    "default": {
        "suggestion": "No specific treatment suggestion available for this condition.",
        "prevention": "General best practices include ensuring proper watering, adequate sunlight, good soil drainage, and regular monitoring."
    }
}

# --- Class names (must match your model) ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Model loader: tries .h5 then SavedModel directory ---
@st.cache_resource
def load_model():
    candidates = ["plant_disease_model.h5", "plant_disease_model_savedmodel"]
    for c in candidates:
        if os.path.exists(c):
            try:
                model = tf.keras.models.load_model(c)
                return model
            except Exception as e:
                print(f"Failed to load model from {c}: {e}")
                continue
    return None

model = load_model()

# --- Session-state defaults ---
if "pump_running" not in st.session_state:
    st.session_state["pump_running"] = False
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "sensor_history" not in st.session_state:
    st.session_state["sensor_history"] = [
        {"moisture": random.randint(40, 70), "temp": round(random.uniform(22, 30), 1), "humidity": random.randint(40, 70)}
        for _ in range(24)
    ]
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📊 About & Dashboard"


# --- Utility: predict safely ---
def predict_image(image):
    if model is None: return None, None, None
    try:
        image = image.convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        idx = int(np.argmax(preds[0]))
        cls = CLASS_NAMES[idx]
        conf = float(np.max(preds[0]) * 100)
        top_idx = np.argsort(preds[0])[-5:][::-1]
        top5 = [(CLASS_NAMES[i], float(preds[0][i] * 100)) for i in top_idx]
        return cls, conf, top5
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# --- UI Starts Here ---
st.markdown("""
    <div style="text-align:center; padding-top: 20px;">
        <h1 style="color:#2e7d32;">🌱 Agro Guard: Smart Crop Care System</h1>
        <h4 style="color:#555;">An AI-powered toolkit for modern farming and gardening</h4>
    </div>
    <hr>
""", unsafe_allow_html=True)

# --- Custom CSS for Radio Buttons to look like Tabs ---
st.markdown("""
<style>
    /* Make radio buttons look like a tab bar */
    div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child{
        background-color: #f0f2f6;
        padding: 10px 20px;
        margin: 0 5px;
        border-radius: 8px;
        border: 1px solid transparent;
        transition: all 0.3s;
    }
    /* Style for the selected tab */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child[aria-checked="true"]{
        background-color: #e0f2fe;
        border: 1px solid #0284c7;
        color: #0c4a6e;
        font-weight: bold;
    }
    /* Hide the actual radio button circle */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div{
        display:none;
    }
</style>
""", unsafe_allow_html=True)

# Main navigation using st.radio to maintain state
active_tab = st.radio(
    "Navigation", 
    ["📊 About & Dashboard", "🔍 Disease Detection", "💧 Smart Watering"], 
    key="nav_radio",
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.active_tab = active_tab # Update session state on selection


# --- About & Dashboard Tab ---
if st.session_state.active_tab == "📊 About & Dashboard":
    st.header("Project Dashboard")
    st.write("""
    **Welcome to Agro Guard!** This project integrates cutting-edge AI with IoT to provide a comprehensive solution for crop care. 
    Below is an overview of the system's features and the technologies used to build it.
    """)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌿 AI Model Classes", len(CLASS_NAMES))
    with col2:
        st.metric("✅ Model Accuracy", "92.98%")
    with col3:
        st.metric("🚀 Project Version", "v3.0")

    st.markdown("---")

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.subheader("🌟 Features Offered")
        st.markdown("""
        - **AI-Powered Disease Diagnosis:** Instantly identify 38 different plant diseases.
        - **Dual Input Modes:** Use either image uploads or your live camera for detection.
        - **Actionable Advice:** Get immediate treatment and prevention suggestions.
        - **IoT Sensor Dashboard:** Monitor real-time (simulated) soil moisture, temperature, and humidity.
        - **Smart Irrigation Control:** Manually activate the watering system and view activity logs.
        """)
    with feat_col2:
        st.subheader("⚙️ Technology Stack")
        st.markdown("""
        - **AI & Machine Learning:** TensorFlow, Keras, NumPy, Pillow
        - **Web Application:** Streamlit
        - **Data Visualization:** Plotly Express
        - **IoT Backend (Planned):** Firebase Realtime Database
        - **Hardware (Planned):** ESP32/NodeMCU, various sensors
        """)
        
# --- Disease Detection Tab ---
if st.session_state.active_tab == "🔍 Disease Detection":
    st.header("Plant Disease Diagnosis")
    st.write("Upload an image of a plant leaf or use your camera to get an instant diagnosis.")
    
    upload_col, camera_col = st.columns(2)
    with upload_col:
        uploaded_file = st.file_uploader("📤 **Upload an image...**", type=["jpg", "jpeg", "png"], key="uploader")
    with camera_col:
        camera_file = st.camera_input("📸 **Take a picture...**", key="camera")
    
    image_source = uploaded_file or camera_file

    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption='Input Image.', use_column_width='always')

        if model is None:
            st.warning("Model not loaded. Cannot perform prediction.")
        else:
            with st.spinner("Analyzing..."):
                cls, conf, top5 = predict_image(image)
            
            if cls:
                st.success(f"**Primary Prediction:** {cls.replace('___', ' - ').replace('_', ' ')}")
                st.info(f"**Confidence:** {conf:.2f}%")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    treatment = treatment_database.get(cls, treatment_database["default"])
                    with st.expander("View Recommended Actions"):
                        st.write("**Treatment:**", treatment["suggestion"])
                        st.write("**Prevention:**", treatment["prevention"])
                
                with res_col2:
                    st.subheader("Top Predictions Breakdown")
                    
                    # --- Improved Pie Chart Logic ---
                    # Take top 3 predictions
                    pie_data = top5[:3]
                    other_confidence = 100 - sum(p[1] for p in pie_data)
                    if other_confidence > 0.1:
                         pie_data.append(("Other", other_confidence))

                    fig = px.pie(
                        values=[p[1] for p in pie_data], 
                        names=[p[0].replace('___', ' - ').replace('_', ' ') for p in pie_data],
                        title='Confidence Distribution',
                        color_discrete_sequence=px.colors.sequential.Greens_r
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not analyze the image. Please try another one.")

# --- Smart Watering Tab ---
if st.session_state.active_tab == "💧 Smart Watering":
    st.header("Smart Watering Dashboard")
    st.write("Monitor your plant's environment and control the irrigation system. (Currently in simulation mode)")

    latest = st.session_state["sensor_history"][-1]
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("💧 **Soil Moisture**", f"{latest['moisture']} %")
    kpi_col2.metric("🌡️ **Temperature**", f"{latest['temp']} °C")
    kpi_col3.metric("💨 **Humidity**", f"{latest['humidity']} %")
    st.markdown("---")
    
    ctrl_col1, ctrl_col2 = st.columns([1.5, 1])
    with ctrl_col1:
        st.subheader("System Control")
        if st.button("▶️ Activate Pump (5s)", disabled=st.session_state["pump_running"]):
            st.session_state["pump_running"] = True
            st.rerun()
    with ctrl_col2:
        st.subheader("Pump Status")
        if st.session_state["pump_running"]:
            with st.spinner("Watering in progress..."):
                start_time = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["logs"].insert(0, f"{start_time} - Pump started manually.")
                for i in range(5):
                    last = st.session_state["sensor_history"][-1]
                    new = {"moisture": min(100, last["moisture"] + 5), "temp": last["temp"], "humidity": min(100, last["humidity"] + 2)}
                    st.session_state["sensor_history"].append(new)
                    time.sleep(1)
                end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["logs"].insert(0, f"{end_time} - Watering completed.")
                st.session_state["pump_running"] = False
            st.success("Watering complete!")
            time.sleep(1) 
            st.rerun() 
        else:
            st.info("⏸️ PUMP IS IDLE")
            
    st.markdown("---")
    
    st.subheader("Sensor Data History (Live)")
    hist = st.session_state["sensor_history"][-48:]
    chart_data = {"Moisture (%)": [p["moisture"] for p in hist], "Temperature (°C)": [p["temp"] for p in hist], "Humidity (%)": [p["humidity"] for p in hist]}
    st.line_chart(chart_data)

    st.markdown("---")
    
    st.subheader("Activity Log")
    log_container = st.container(height=200)
    with log_container:
        if not st.session_state["logs"]:
            st.info("No activity recorded yet.")
        else:
            for entry in st.session_state["logs"]:
                st.text(f" - {entry}")

