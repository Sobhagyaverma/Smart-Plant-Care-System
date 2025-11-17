import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import random
import os
import plotly.express as px
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db


# --- App Configuration ---
st.set_page_config(page_title="Agro Guard: Smart Plant Care System", layout="wide", initial_sidebar_state="collapsed")

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

@st.cache_resource
def init_firebase():
    try:
        firebase_admin.get_app()
    except ValueError:
        try:
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://agro-guard-iot-system-default-rtdb.asia-southeast1.firebasedatabase.app' 
            })
        except Exception as e:
            st.error(f"Error initializing Firebase: {e}")
            st.error("Please make sure 'firebase-key.json' is in the correct folder and your Database URL is correct.")
            return False
    return True

firebase_ready = init_firebase()

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

# --- CUSTOM CSS FOR PROFESSIONAL DARK THEME ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #e0f2fe;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card containers */
    .dashboard-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-card h3 {
        color: #10b981;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #10b981;
        padding-bottom: 0.5rem;
    }
    
    .dashboard-card p, .dashboard-card li {
        color: #e2e8f0;
        line-height: 1.8;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 1rem !important;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #fbbf24 !important;
    }
    
    /* Tab navigation */
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: center;
        gap: 1rem;
        background: rgba(30, 41, 59, 0.6);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background: rgba(51, 65, 85, 0.8);
        padding: 12px 24px;
        margin: 0;
        border-radius: 8px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        color: #cbd5e1;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child:hover {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.2);
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child[aria-checked="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: 2px solid #10b981;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div {
        display: none;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }
    
    .stButton > button:disabled {
        background: #475569;
        box-shadow: none;
    }
    
    /* File uploader and camera */
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] {
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed #10b981;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(30, 41, 59, 0.8) !important;
        border-left: 4px solid #10b981 !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
    }
    
    .stWarning {
        border-left-color: #fbbf24 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(51, 65, 85, 0.6);
        border-radius: 8px;
        color: #10b981 !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #e2e8f0;
    }
    
    /* Charts */
    .js-plotly-plot {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 12px;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p, span, div {
        color: #cbd5e1;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-live {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-sim {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    /* Activity log */
    .log-container {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 1rem;
        max-height: 200px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        color: #10b981;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #10b981;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #059669;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>🌱 Agro Guard</h1>
        <p>Smart Crop Care System - AI-Powered Agriculture</p>
    </div>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
active_tab = st.radio(
    "Navigation", 
    ["📊 About & Dashboard", "🔍 Disease Detection", "💧 Smart Watering"], 
    key="nav_radio",
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.active_tab = active_tab

# ============================
# 📊 ABOUT & DASHBOARD TAB
# ============================
if st.session_state.active_tab == "📊 About & Dashboard":
    
    # Project at a Glance
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Project at a Glance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌿 AI Model Classes", len(CLASS_NAMES))
    with col2:
        st.metric("✅ Model Accuracy", "92.98%")
    with col3:
        st.metric("🚀 Project Version", "v3.0")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Two column layout for Features and Tech Stack
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 Key Features")
        st.markdown("""
        - **AI-Powered Disease Diagnosis** - Instantly identify 38 different plant diseases using deep learning
        - **Dual Input Modes** - Use image uploads or live camera for real-time detection
        - **Actionable Insights** - Get immediate treatment and prevention suggestions
        - **IoT Sensor Dashboard** - Monitor real-time soil moisture, temperature, and humidity
        - **Smart Irrigation Control** - Automated watering system with activity logging
        - **Firebase Integration** - Live data sync with cloud database
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Technology Stack")
        st.markdown("""
        **AI & Machine Learning**
        - TensorFlow 2.x & Keras
        - NumPy & Pillow (PIL)
        
        **Web Application**
        - Streamlit Framework
        - Plotly Express & Plotly GO
        
        **IoT Backend**
        - Firebase Realtime Database
        - Firebase Admin SDK
        
        **Hardware** (Planned)
        - ESP32/NodeMCU Microcontrollers
        - DHT22 Sensors, Soil Moisture Sensors
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 How It Works")
    st.markdown("""
    **Step 1: Image Capture** → User uploads or captures a plant leaf image  
    **Step 2: AI Processing** → TensorFlow model analyzes the image using CNN architecture  
    **Step 3: Disease Identification** → System predicts disease with confidence scores  
    **Step 4: Treatment Recommendations** → Displays actionable treatment and prevention advice  
    **Step 5: IoT Monitoring** → Real-time sensor data from Firebase guides irrigation decisions  
    **Step 6: Smart Watering** → Automated pump control based on soil moisture levels
    """)
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

# ============================
# 🔍 DISEASE DETECTION TAB
# ============================
elif st.session_state.active_tab == "🔍 Disease Detection":
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Plant Disease Diagnosis")
    st.markdown("Upload an image of a plant leaf or use your camera to get an instant AI-powered diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input methods
    col_upload, col_camera = st.columns(2)
    with col_upload:
        uploaded_file = st.file_uploader("📤 **Upload an image...**", type=["jpg", "jpeg", "png"], key="uploader")
    with col_camera:
        camera_file = st.camera_input("📸 **Take a picture...**", key="camera")
    
    image_source = uploaded_file or camera_file

    if image_source is not None:
        image = Image.open(image_source)
        
        # Display image in a card
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.image(image, caption='Input Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if model is None:
            st.warning("⚠️ Model not loaded. Cannot perform prediction.")
        else:
            with st.spinner("🔄 Analyzing image with AI..."):
                cls, conf, top5 = predict_image(image)
            
            if cls:
                # Results section
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                
                # Primary prediction with confidence
                disease_name = cls.replace('___', ' - ').replace('_', ' ')
                st.success(f"**🎯 Primary Prediction:** {disease_name}")
                st.info(f"**📊 Confidence:** {conf:.2f}%")
                
                # Progress bar for confidence
                st.progress(conf/100)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Two column layout for treatment and chart
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    treatment = treatment_database.get(cls, treatment_database["default"])
                    st.markdown("### 💊 Recommended Actions")
                    
                    st.markdown("**Treatment:**")
                    st.markdown(f"_{treatment['suggestion']}_")
                    
                    st.markdown("**Prevention:**")
                    st.markdown(f"_{treatment['prevention']}_")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Confidence Distribution")
                    
                    # Enhanced pie chart
                    pie_data = top5[:3]
                    other_confidence = 100 - sum(p[1] for p in pie_data)
                    if other_confidence > 0.1:
                        pie_data.append(("Other", other_confidence))

                    fig = px.pie(
                        values=[p[1] for p in pie_data], 
                        names=[p[0].replace('___', ' - ').replace('_', ' ') for p in pie_data],
                        color_discrete_sequence=['#10b981', '#059669', '#047857', '#065f46']
                    )
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        textfont=dict(size=11, color='white'),
                        marker=dict(line=dict(color='#1e293b', width=2))
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        showlegend=True,
                        legend=dict(
                            bgcolor='rgba(30, 41, 59, 0.8)',
                            bordercolor='#10b981',
                            borderwidth=1
                        ),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Top 5 predictions table
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown("### 🏆 Top 5 Predictions")
                
                for i, (name, prob) in enumerate(top5, 1):
                    col_rank, col_name, col_prob = st.columns([0.5, 3, 1])
                    with col_rank:
                        st.markdown(f"**#{i}**")
                    with col_name:
                        st.markdown(name.replace('___', ' - ').replace('_', ' '))
                    with col_prob:
                        st.markdown(f"{prob:.2f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Could not analyze the image. Please try another one.")

# ============================
# 💧 SMART WATERING TAB
# ============================
elif st.session_state.active_tab == "💧 Smart Watering":
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 💧 Smart Irrigation Dashboard")
    st.markdown("Monitor your plant environment and control irrigation. System automatically switches between Live and Simulation modes.")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # 1️⃣ TRY TO READ LIVE DATA FROM FIREBASE
    # -------------------------------------------------------
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

    # Status badge
    if live_mode:
        st.markdown('<span class="status-badge status-live">🟢 LIVE MODE: Firebase IoT Connected</span>', unsafe_allow_html=True)
        moisture = live_data.get("moisture", 0)
        temperature = live_data.get("temperature", 0)
        humidity = live_data.get("humidity", 0)

        # Moisture alert logic
        if moisture < 30:
            moisture_delta = "Very Dry!"
        elif moisture < 50:
            moisture_delta = "Dry"
        else:
            moisture_delta = "Normal"
    else:
        st.markdown('<span class="status-badge status-sim">🟡 SIMULATION MODE: Firebase Not Connected</span>', unsafe_allow_html=True)
        latest = st.session_state["sensor_history"][-1]
        moisture = latest["moisture"]
        temperature = latest["temp"]
        humidity = latest["humidity"]
        moisture_delta = ""

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # 2️⃣ KPI METRICS
    # -------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Environmental Sensors")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("💧 Soil Moisture", f"{moisture}%", delta=moisture_delta)
    k2.metric("🌡️ Temperature", f"{temperature}°C")
    k3.metric("💨 Humidity", f"{humidity}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # 3️⃣ PUMP CONTROL
    # -------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 🎮 System Control")

    if live_mode:
        if st.button("▶️ Activate Pump (Firebase)", use_container_width=True):
            try:
                pump_ref = db.reference("/iot_dashboard/controls")
                pump_ref.set({"pump": True})

                log_ref = db.reference("/iot_dashboard/logs")
                log_ref.set({"last_event": "Manual watering activated from web."})

                st.success("✅ Pump command sent to ESP32 successfully!")
            except Exception as e:
                st.error(f"❌ Error sending pump command: {e}")
    else:
        # Simulation pump
        if st.button("▶️ Activate Pump (Simulated)", disabled=st.session_state["pump_running"], use_container_width=True):
            st.session_state["pump_running"] = True
            st.rerun()

        if st.session_state["pump_running"]:
            with st.spinner("💦 Watering in progress..."):
                start = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["logs"].insert(0, f"{start} - Pump started.")

                # Simulate watering
                for i in range(5):
                    last = st.session_state["sensor_history"][-1]
                    new = {
                        "moisture": min(100, last["moisture"] + 5),
                        "temp": last["temp"],
                        "humidity": min(100, last["humidity"] + 2)
                    }
                    st.session_state["sensor_history"].append(new)
                    time.sleep(1)

                end = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["logs"].insert(0, f"{end} - Watering completed.")

            st.session_state["pump_running"] = False
            st.success("✅ Watering complete!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # 4️⃣ SENSOR HISTORY CHART
    # -------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Sensor History")

    if live_mode:
        # Create enhanced chart for live mode
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=moisture,
            domain={'x': [0, 0.3], 'y': [0, 1]},
            title={'text': "Moisture %"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#10b981"},
                   'threshold': {
                       'line': {'color': "red", 'width': 4},
                       'thickness': 0.75,
                       'value': 30}}))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=temperature,
            domain={'x': [0.35, 0.65], 'y': [0, 1]},
            title={'text': "Temperature °C"},
            gauge={'axis': {'range': [0, 50]},
                   'bar': {'color': "#f59e0b"}}))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=humidity,
            domain={'x': [0.7, 1], 'y': [0, 1]},
            title={'text': "Humidity %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#3b82f6"}}))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=14),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Line chart for simulation mode
        hist = st.session_state["sensor_history"][-48:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[p["moisture"] for p in hist],
            name="Moisture (%)",
            line=dict(color='#10b981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            y=[p["temp"] for p in hist],
            name="Temperature (°C)",
            line=dict(color='#f59e0b', width=3)
        ))
        fig.add_trace(go.Scatter(
            y=[p["humidity"] for p in hist],
            name="Humidity (%)",
            line=dict(color='#3b82f6', width=3)
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.2)',
                title="Time Steps"
            ),
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.2)',
                title="Value"
            ),
            legend=dict(
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='#10b981',
                borderwidth=1
            ),
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # 5️⃣ ACTIVITY LOG
    # -------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Activity Log")

    if live_mode:
        try:
            log_ref = db.reference("/iot_dashboard/logs/last_event")
            last_log = log_ref.get()

            if last_log:
                st.markdown(f'<div class="log-container">📌 {last_log}</div>', unsafe_allow_html=True)
            else:
                st.info("No events recorded yet.")
        except Exception as e:
            st.error(f"❌ Error reading logs: {e}")
    else:
        if not st.session_state["logs"]:
            st.info("No activity recorded yet.")
        else:
            log_html = '<div class="log-container">'
            for entry in st.session_state["logs"][:10]:  # Show last 10 entries
                log_html += f"• {entry}<br>"
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem; border-top: 1px solid rgba(148, 163, 184, 0.2);">
        <p>🌱 Agro Guard v3.0 | Powered by AI & IoT | © 2024</p>
    </div>
""", unsafe_allow_html=True)