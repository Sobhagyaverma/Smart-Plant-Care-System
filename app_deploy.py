import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import random
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Deployment Version - IoT Simulation Mode (No Firebase Required)


# --- App Configuration ---
st.set_page_config(
    page_title="Agro Guard: Smart Plant Care System", 
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Agro Guard v3.0 - AI-Powered Plant Care System"
    }
)

# --- Initialize session state for sensor history with timestamps ---
if "sensor_history" not in st.session_state:
    import datetime
    now = datetime.datetime.now()
    st.session_state["sensor_history"] = [
        {
            "moisture": random.randint(40, 70), 
            "temp": round(random.uniform(22, 30), 1), 
            "humidity": random.randint(40, 70),
            "timestamp": (now - datetime.timedelta(hours=48-i)).strftime("%H:%M")
        }
        for i in range(48)
    ]

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0;
    }
    
    /* Header Animation */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 60px 40px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 30px;
        animation: fadeInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    .hero-section h1 {
        color: white;
        font-size: 3.5em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-section h4 {
        color: #dbeafe;
        font-size: 1.3em;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Navigation Tabs */
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: center;
        gap: 15px;
        padding: 20px 0;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background: #1e293b;
        padding: 15px 30px;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 2px solid transparent;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        border-color: #3b82f6;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] {
        font-size: 1.1em;
        font-weight: 600;
        color: #cbd5e1;
    }
    
    /* Selected Tab */
    div.row-widget.stRadio > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        border-color: #1e40af;
        transform: translateY(-5px);
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Cards */
    .feature-card {
        background: #1e293b;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        height: 100%;
        border: 1px solid #334155;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.3);
        border-color: #3b82f6;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 50px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.5);
    }
    
    .stButton > button:disabled {
        background: #475569;
        cursor: not-allowed;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 10px;
        font-weight: 600;
        color: #cbd5e1;
        border: 1px solid #334155;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Section Container */
    .content-section {
        background: #1e293b;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        margin: 20px 0;
        border: 1px solid #334155;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #e2e8f0;
    }
    
    /* Plotly dark theme */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

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

# --- Class names (from labels.txt) ---
try:
    with open("labels.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("labels.txt not found. Please ensure it exists.")
    CLASS_NAMES = []
except Exception as e:
    st.error(f"Error reading labels.txt: {e}")
    CLASS_NAMES = []

# --- Model loader: Hybrid CNN (Feature Extractor) + SVM ---
@st.cache_resource
def load_model_pipeline():
    try:
        # Load CNN (MobileNetV2)
        base_model_path = "cnn_mobilenetv2.keras"
        if not os.path.exists(base_model_path):
            st.error(f"CNN model not found at {base_model_path}")
            return None, None, None
        
        full_cnn = tf.keras.models.load_model(base_model_path)
        
        # User requested: cnn.layers[-3].output
        feature_extractor = tf.keras.Model(inputs=full_cnn.input, outputs=full_cnn.layers[-3].output)
        
        # Load Scaler
        scaler = joblib.load("scaler.pkl")
        
        # Load SVM
        svm_model = joblib.load("svm_linear.pkl")
        
        return feature_extractor, scaler, svm_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

feature_extractor, scaler, svm_model = load_model_pipeline()

# --- Deployment Version: IoT Simulation Mode (No Firebase) ---
# Firebase is disabled for deployment - using simulation mode only
firebase_ready = False  # Always False for deployment

# --- Session-state defaults ---
if "pump_running" not in st.session_state:
    st.session_state["pump_running"] = False
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📊 About & Dashboard"

# --- Utility: predict safely with Hybrid Pipeline ---
def predict_image(image):
    if feature_extractor is None or scaler is None or svm_model is None:
        return None, None, None
    try:
        # 1. Preprocess
        image = image.convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
        # Use MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # 2. Extract Features
        features = feature_extractor.predict(img_array, verbose=0)
        
        # 3. Scale Features
        features_scaled = scaler.transform(features)
        
        # 4. Predict with SVM
        # Since LinearSVC doesn't have predict_proba, use decision_function + softmax approximation
        decision_scores = svm_model.decision_function(features_scaled)[0]
        
        # Softmax to get probabilities
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        if exp_scores.sum() == 0:
             probs = exp_scores 
        else:
             probs = exp_scores / exp_scores.sum()
        
        # Get Top 1
        idx = int(np.argmax(probs))
         # Ensure idx is within bounds of CLASS_NAMES
        if idx < len(CLASS_NAMES):
            cls = CLASS_NAMES[idx]
        else:
            cls = "Unknown"
            
        conf = float(probs[idx] * 100)
        
        # Get Top 5
        top_idx = np.argsort(probs)[-5:][::-1]
        top5 = []
        for i in top_idx:
            if i < len(CLASS_NAMES):
                 top5.append((CLASS_NAMES[i], float(probs[i] * 100)))
                 
        return cls, conf, top5
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# --- Hero Section ---
st.markdown("""
    <div class="hero-section">
        <h1>🌱 Agro Guard</h1>
        <h4>Smart Plant Care System Powered by AI & IoT</h4>
    </div>
""", unsafe_allow_html=True)

# Main navigation
active_tab = st.radio(
    "Navigation", 
    ["📊 About & Dashboard", "🔍 Disease Detection", "💧 Smart Watering"], 
    key="nav_radio",
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.active_tab = active_tab

# --- About & Dashboard Tab ---
if st.session_state.active_tab == "📊 About & Dashboard":
    
    # Quick Stats Section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌿 Plant Classes", len(CLASS_NAMES), delta="38 Total")
    with col2:
        st.metric("✅ Hybrid AI", "95.85%", delta="CNN+SVM")
    with col3:
        st.metric("🚀 Model", "MobileNetV2", delta="+ Linear SVM")
    with col4:
        st.metric("⚡ Status", "Active", delta="Online")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Content
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("### 🌟 Key Features")
        
        features = [
            {"icon": "🤖", "title": "Hybrid AI Diagnosis", "desc": "CNN feature extraction + SVM classification with 95.85% accuracy"},
            {"icon": "📸", "title": "Dual Input Modes", "desc": "Upload images or use live camera for real-time detection"},
            {"icon": "💊", "title": "Treatment Recommendations", "desc": "Get actionable treatment and prevention strategies"},
            {"icon": "📊", "title": "IoT Dashboard", "desc": "Monitor soil moisture, temperature, and humidity in real-time"},
            {"icon": "💧", "title": "Smart Irrigation", "desc": "Automated watering control with activity logging"},
            {"icon": "📱", "title": "Responsive Design", "desc": "Works seamlessly on desktop, tablet, and mobile devices"}
        ]
        
        for feature in features:
            st.markdown(f"""
            <div style="padding: 15px; margin: 10px 0; background: linear-gradient(135deg, #334155 0%, #1e293b 100%); 
                        border-radius: 10px; border-left: 4px solid #3b82f6;">
                <div style="font-size: 1.5em; margin-bottom: 5px; color: #e2e8f0;">{feature['icon']} <strong>{feature['title']}</strong></div>
                <div style="color: #94a3b8; font-size: 0.95em;">{feature['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Technology Stack")
        
        tech_stack = {
            "AI & Machine Learning": ["TensorFlow", "Keras", "NumPy", "Pillow"],
            "Web Framework": ["Streamlit"],
            "Data Visualization": ["Plotly Express"],
            "IoT Backend": ["Firebase Realtime DB"],
            "Hardware": ["ESP32/NodeMCU", "Soil Sensors", "DHT22"]
        }
        
        for category, technologies in tech_stack.items():
            st.markdown(f"**{category}**")
            tech_tags = " ".join([f'<span style="background: #334155; color: #60a5fa; padding: 5px 15px; border-radius: 20px; margin: 5px; display: inline-block; font-size: 0.9em; border: 1px solid #475569;">{tech}</span>' for tech in technologies])
            st.markdown(tech_tags, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Health Gauge
        st.markdown('<div class="content-section" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown("### 🎯 System Health")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 95.85,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Hybrid CNN+SVM Performance", 'font': {'size': 18, 'color': '#e2e8f0'}},
            delta = {'reference': 90, 'increasing': {'color': "#3b82f6"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#3b82f6"},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#475569",
                'steps': [
                    {'range': [0, 50], 'color': '#7f1d1d'},
                    {'range': [50, 75], 'color': '#854d0e'},
                    {'range': [75, 100], 'color': '#065f46'}],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.75,
                    'value': 95}}))
        
        fig.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 🔬 How It Works")
    
    process_cols = st.columns(4)
    steps = [
        {"num": "1", "title": "Capture", "icon": "📸", "desc": "Upload or take a photo of the plant leaf"},
        {"num": "2", "title": "Analyze", "icon": "🤖", "desc": "AI model processes the image using deep learning"},
        {"num": "3", "title": "Diagnose", "icon": "🔍", "desc": "System identifies disease with confidence score"},
        {"num": "4", "title": "Recommend", "icon": "💊", "desc": "Get treatment and prevention suggestions"}
    ]
    
    for idx, step in enumerate(steps):
        with process_cols[idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #334155 0%, #1e293b 100%); 
                        border-radius: 15px; margin: 10px 0; border: 1px solid #475569;">
                <div style="font-size: 3em; margin-bottom: 10px;">{step['icon']}</div>
                <div style="font-size: 1.8em; color: #3b82f6; font-weight: 700; margin-bottom: 5px;">{step['num']}</div>
                <div style="font-size: 1.2em; font-weight: 600; color: #60a5fa; margin-bottom: 10px;">{step['title']}</div>
                <div style="font-size: 0.9em; color: #94a3b8;">{step['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

     # System Architecture Diagram
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 🏗️ System Architecture")
    
    # Create architecture diagram
    fig = go.Figure()
    
    # Define nodes
    nodes = {
        'User Interface': (0.5, 0.9),
        'Streamlit App': (0.5, 0.7),
        'TensorFlow Model': (0.2, 0.4),
        'Firebase DB': (0.8, 0.4),
        'IoT Sensors': (0.5, 0.1)
    }
    
    # Add nodes
    for name, (x, y) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color='#10b981', line=dict(width=2, color='white')),
            text=name,
            textposition='bottom center',
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            hovertext=name
        ))
    
    # Add connections
    connections = [
        ('User Interface', 'Streamlit App'),
        ('Streamlit App', 'TensorFlow Model'),
        ('Streamlit App', 'Firebase DB'),
        ('Firebase DB', 'IoT Sensors')
    ]
    
    for start, end in connections:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='#10b981', width=2),
            hoverinfo='none'
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.1]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- Disease Detection Tab ---
elif st.session_state.active_tab == "🔍 Disease Detection":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 🔬 Plant Disease Diagnosis")
    st.markdown("Upload an image of a plant leaf or use your camera for instant AI-powered diagnosis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    upload_col, camera_col = st.columns(2)
    with upload_col:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="uploader", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with camera_col:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("#### 📸 Take Photo")
        camera_file = st.camera_input("Capture image", key="camera", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    image_source = uploaded_file or camera_file

    if image_source is not None:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        image = Image.open(image_source)
        
        # Display image with styling
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        col_img, col_space = st.columns([2, 1])
        with col_img:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if feature_extractor is None:
            st.warning("⚠️ Models not loaded. Cannot perform prediction.")
        else:
            with st.spinner("🔄 Analyzing image with AI..."):
                time.sleep(0.5)  # Brief pause for better UX
                cls, conf, top5 = predict_image(image)
            
            if cls:
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
                
                # Results Section
                st.markdown('<div class="content-section">', unsafe_allow_html=True)
                st.markdown("### 📊 Diagnosis Results")
                
                # Primary Result with enhanced styling
                is_healthy = "healthy" in cls.lower()
                result_color = "#10b981" if is_healthy else "#ef4444"
                result_bg = "#064e3b" if is_healthy else "#7f1d1d"
                
                st.markdown(f"""
                <div style="background: {result_bg}; padding: 25px; border-radius: 15px; 
                            border-left: 6px solid {result_color}; margin: 20px 0;">
                    <div style="font-size: 1.1em; color: #94a3b8; margin-bottom: 10px;">Primary Diagnosis</div>
                    <div style="font-size: 2em; font-weight: 700; color: {result_color}; margin-bottom: 10px;">
                        {cls.replace('___', ' - ').replace('_', ' ')}
                    </div>
                    <div style="font-size: 1.3em; color: #e2e8f0;">
                        Confidence: <strong>{conf:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Treatment and Charts
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1, 1], gap="large")
                
                with res_col1:
                    st.markdown('<div class="content-section">', unsafe_allow_html=True)
                    st.markdown("### 💊 Recommended Actions")
                    
                    treatment = treatment_database.get(cls, treatment_database["default"])
                    
                    st.markdown(f"""
                    <div style="background: #422006; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #78350f;">
                        <div style="font-size: 1.2em; font-weight: 600; color: #fbbf24; margin-bottom: 10px;">
                            🔧 Treatment
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.6;">
                            {treatment["suggestion"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: #0c4a6e; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #075985;">
                        <div style="font-size: 1.2em; font-weight: 600; color: #38bdf8; margin-bottom: 10px;">
                            🛡️ Prevention
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.6;">
                            {treatment["prevention"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with res_col2:
                    st.markdown('<div class="content-section">', unsafe_allow_html=True)
                    st.markdown("### 📈 Confidence Distribution")
                    
                    # Enhanced Pie Chart
                    pie_data = top5[:3]
                    other_confidence = 100 - sum(p[1] for p in pie_data)
                    if other_confidence > 0.1:
                        pie_data.append(("Other", other_confidence))

                    fig = go.Figure(data=[go.Pie(
                        labels=[p[0].replace('___', ' - ').replace('_', ' ') for p in pie_data],
                        values=[p[1] for p in pie_data],
                        hole=.4,
                        marker=dict(colors=['#3b82f6', '#60a5fa', '#93c5fd', '#475569']),
                        textinfo='label+percent',
                        textposition='outside',
                        textfont=dict(size=11, color='#e2e8f0')
                    )])
                    
                    fig.update_layout(
                        showlegend=True,
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1, font=dict(color='#e2e8f0')),
                        paper_bgcolor='#1e293b',
                        plot_bgcolor='#1e293b'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top 5 Predictions Bar
                    st.markdown("#### 🎯 Top 5 Predictions")
                    for i, (pred_cls, pred_conf) in enumerate(top5, 1):
                        bar_color = "#3b82f6" if i == 1 else "#60a5fa"
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 3px;">
                                {i}. {pred_cls.replace('___', ' - ').replace('_', ' ')}
                            </div>
                            <div style="background: #334155; border-radius: 10px; overflow: hidden;">
                                <div style="background: {bar_color}; width: {pred_conf}%; padding: 5px 10px; 
                                            color: white; font-weight: 600; font-size: 0.85em; border-radius: 10px;">
                                    {pred_conf:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Could not analyze the image. Please try another one.")

# --- Smart Watering Tab ---
elif st.session_state.active_tab == "💧 Smart Watering":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 💧 Smart Watering Dashboard")
    st.markdown("Monitor your plant environment and control irrigation. Automatically switches between Firebase Live Mode & Simulation Mode.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Try to read live data from Firebase
    live_mode = False
    live_data = None

    if firebase_ready:
        try:
            live_data_ref = db.reference("/iot_dashboard/live_data")
            live_data = live_data_ref.get()
            if live_data:
                live_mode = True
            else:
                live_mode = False
        except:
            live_mode = False

    # Mode Indicator
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    if live_mode:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); padding: 15px; 
                    border-radius: 10px; border-left: 5px solid #10b981; text-align: center; border: 1px solid #059669;">
            <span style="font-size: 1.3em; font-weight: 600; color: #6ee7b7;">
                🟢 LIVE MODE: Firebase IoT Connected
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        moisture = live_data.get("moisture", 0)
        temperature = live_data.get("temperature", 0)
        humidity = live_data.get("humidity", 0)

        if moisture < 30:
            moisture_delta = "Very Dry!"
        elif moisture < 50:
            moisture_delta = "Dry"
        else:
            moisture_delta = "Normal"
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #78350f 0%, #92400e 100%); padding: 15px; 
                    border-radius: 10px; border-left: 5px solid #fbbf24; text-align: center; border: 1px solid #b45309;">
            <span style="font-size: 1.3em; font-weight: 600; color: #fde68a;">
                🟡 SIMULATION MODE: Pump Not Connected
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        latest = st.session_state["sensor_history"][-1]
        moisture = latest["moisture"]
        temperature = latest["temp"]
        humidity = latest["humidity"]
        moisture_delta = ""

    # KPI Metrics with enhanced design
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    
    with k1:
        st.markdown('<div class="content-section" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 3em; margin-bottom: 10px;">💧</div>
        <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 5px;">Soil Moisture</div>
        <div style="font-size: 2.5em; font-weight: 700; color: #3b82f6;">{moisture}%</div>
        <div style="font-size: 0.9em; color: #ef4444; font-weight: 600;">{moisture_delta}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with k2:
        st.markdown('<div class="content-section" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 3em; margin-bottom: 10px;">🌡️</div>
        <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 5px;">Temperature</div>
        <div style="font-size: 2.5em; font-weight: 700; color: #f59e0b;">{temperature}°C</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with k3:
        st.markdown('<div class="content-section" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 3em; margin-bottom: 10px;">💨</div>
        <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 5px;">Humidity</div>
        <div style="font-size: 2.5em; font-weight: 700; color: #8b5cf6;">{humidity}%</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

    # Pump Control
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Irrigation Control")
    
    control_col1, control_col2 = st.columns([1, 2])
    
    with control_col1:
        if live_mode:
            if st.button("▶️ Activate Pump", use_container_width=True, type="primary"):
                try:
                    pump_ref = db.reference("/iot_dashboard/controls")
                    pump_ref.set({"pump": True})

                    log_ref = db.reference("/iot_dashboard/logs")
                    log_ref.set({"last_event": "Manual watering activated from web."})

                    st.success("✅ Pump command sent to ESP32 successfully!")
                except Exception as e:
                    st.error(f"❌ Error sending pump command: {e}")
        else:
            if st.button("▶️ Activate Pump (Simulated)", disabled=st.session_state["pump_running"], use_container_width=True, type="primary"):
                st.session_state["pump_running"] = True
                st.rerun()

            if st.session_state["pump_running"]:
                with st.spinner("💧 Watering in progress..."):
                    import datetime
                    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["logs"].insert(0, f"{start} - Pump started.")

                    for i in range(5):
                        last = st.session_state["sensor_history"][-1]
                        now = datetime.datetime.now()
                        new = {
                            "moisture": min(100, last["moisture"] + 5),
                            "temp": round(last["temp"] + random.uniform(-0.2, 0.2), 1),
                            "humidity": min(100, last["humidity"] + 2),
                            "timestamp": now.strftime("%H:%M")
                        }
                        st.session_state["sensor_history"].append(new)
                        time.sleep(1)

                    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["logs"].insert(0, f"{end} - Watering completed.")

                st.session_state["pump_running"] = False
                st.success("✅ Watering complete!")
                st.rerun()
    
    with control_col2:
        st.markdown("""
        <div style="background: #0c4a6e; padding: 20px; border-radius: 10px; border: 1px solid #075985;">
            <div style="font-weight: 600; color: #38bdf8; margin-bottom: 10px;">💡 Quick Tips</div>
            <ul style="color: #cbd5e1; line-height: 1.8; margin: 0;">
                <li>Water when soil moisture drops below 40%</li>
                <li>Avoid watering during peak heat hours</li>
                <li>Monitor for at least 5 minutes after activation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Sensor History Chart
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Sensor Data History")

    if live_mode:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=moisture,
            title={'text': "Moisture", 'font': {'color': '#e2e8f0'}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#3b82f6"},
                   'bgcolor': "#1e293b",
                   'steps': [
                       {'range': [0, 30], 'color': "#7f1d1d"},
                       {'range': [30, 70], 'color': "#854d0e"},
                       {'range': [70, 100], 'color': "#065f46"}]},
            domain={'x': [0, 0.3], 'y': [0, 1]}))
        
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=temperature,
            title={'text': "Temperature (°C)", 'font': {'color': '#e2e8f0'}},
            gauge={'axis': {'range': [0, 50]},
                   'bar': {'color': "#f59e0b"},
                   'bgcolor': "#1e293b"},
            domain={'x': [0.35, 0.65], 'y': [0, 1]}))
        
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=humidity,
            title={'text': "Humidity", 'font': {'color': '#e2e8f0'}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#8b5cf6"},
                   'bgcolor': "#1e293b"},
            domain={'x': [0.7, 1], 'y': [0, 1]}))
        
        fig.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        hist = st.session_state["sensor_history"][-48:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p["timestamp"] for p in hist],
            y=[p["moisture"] for p in hist],
            name="Moisture (%)",
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=[p["timestamp"] for p in hist],
            y=[p["temp"] for p in hist],
            name="Temperature (°C)",
            line=dict(color='#f59e0b', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[p["timestamp"] for p in hist],
            y=[p["humidity"] for p in hist],
            name="Humidity (%)",
            line=dict(color='#8b5cf6', width=3)
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified',
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                font=dict(color='#e2e8f0')
            ),
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            xaxis=dict(
                showgrid=True,
                gridcolor='#334155',
                color='#94a3b8'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#334155',
                color='#94a3b8'
            ),
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Activity Log
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Activity Log")

    if live_mode:
        try:
            log_ref = db.reference("/iot_dashboard/logs/last_event")
            last_log = log_ref.get()

            if last_log:
                st.markdown(f"""
                <div style="background: #f5f5f5; padding: 15px; border-radius: 10px; 
                            border-left: 4px solid #2e7d32;">
                    <div style="color: #666; font-size: 0.9em; margin-bottom: 5px;">Latest Event</div>
                    <div style="color: #424242; font-weight: 600;">📌 {last_log}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No events recorded yet.")
        except Exception as e:
            st.error(f"❌ Error reading logs: {e}")
    else:
        if not st.session_state["logs"]:
            st.info("ℹ️ No activity recorded yet.")
        else:
            log_container = st.container(height=250)
            with log_container:
                for entry in st.session_state["logs"]:
                    st.markdown(f"""
                    <div style="background: #fafafa; padding: 10px 15px; margin: 5px 0; 
                                border-radius: 8px; border-left: 3px solid #2e7d32;">
                        • {entry}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%); 
            border-radius: 20px; color: white;">
    <div style="font-size: 1.5em; font-weight: 600; margin-bottom: 10px;">🌱 Agro Guard</div>
    <div style="font-size: 0.9em; opacity: 0.9;">Smart Plant Care System | Version 3.0</div>
    <div style="font-size: 0.85em; margin-top: 10px; opacity: 0.8;">
        Powered by AI & IoT | Built with ❤️ for Modern Farming
    </div>
</div>
""", unsafe_allow_html=True)