import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration with Dark Theme
st.set_page_config(
    page_title="CHEST X-RAY 5 CLASSES CLASSIFIER",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling - Black, Green, Yellow, White Theme
st.markdown("""
    <style>
    /* Overall Background */
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    /* Main container */
    .main {
        background-color: #0a0a0a;
        padding: 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 3px solid #22c55e;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #22c55e;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(34, 197, 94, 0.3);
    }
    
    /* Title styling */
    [data-testid="stHeadingContainer"] {
        border-bottom: 3px solid #eab308;
        padding-bottom: 20px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: #000000;
        border: 2px solid #eab308;
        border-radius: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%);
        color: #000000;
        box-shadow: 0 8px 20px rgba(234, 179, 8, 0.5);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: #1a1a1a;
        border: 2px dashed #22c55e;
        border-radius: 12px;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #22c55e;
    }
    
    .stAlert[data-baseweb="notification"] {
        background-color: #1a1a1a;
        border-left: 5px solid #22c55e;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 15px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-image: linear-gradient(to right, #22c55e, #eab308);
    }
    
    /* Dividers */
    hr {
        border-color: #22c55e;
        border-width: 2px;
    }
    
    /* Text input fields */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #22c55e;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #eab308;
        box-shadow: 0 0 10px rgba(234, 179, 8, 0.3);
    }
    
    /* Cards/Sections */
    .card {
        background-color: #1a1a1a;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
    
    /* Prediction Result Box */
    .result-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f3a1f 100%);
        border: 2px solid #22c55e;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.2);
    }
    
    /* Confidence Score Box */
    .confidence-box {
        background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%);
        color: #000000;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(234, 179, 8, 0.4);
    }
    
    /* Class probability container */
    .prob-container {
        background-color: #1a1a1a;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #0f3a1f 100%); 
                   border: 3px solid #22c55e; border-radius: 15px; padding: 40px; text-align: center;
                   box-shadow: 0 12px 35px rgba(34, 197, 94, 0.3);'>
            <h1 style='margin: 0; color: #eab308; font-size: 3.2em; font-weight: 900; letter-spacing: 1px;'>🩻 CHEST X-RAY</h1>
            <h2 style='margin: 8px 0; color: #22c55e; font-size: 2.8em; font-weight: 900;'>5 CLASSES CLASSIFIER</h2>
            <div style='background: linear-gradient(90deg, #22c55e, #eab308); height: 3px; border-radius: 2px; margin: 15px 0;'></div>
            <p style='color: #eab308; margin: 15px 0 0 0; font-size: 1.2em; font-weight: bold;'>AI-Powered Diagnostic Assistant</p>
            <p style='color: #ffffff; margin: 8px 0 0 0; font-size: 0.95em;'>🤖 Advanced DenseNet-121 Classification</p>
            <p style='color: #22c55e; margin: 5px 0 0 0; font-size: 0.85em;'>Normal • COVID-19 • Bacterial Pneumonia • Viral Pneumonia • Tuberculosis</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 4. Sidebar Configuration
with st.sidebar:
    st.markdown("""
        <h2 style='color: #eab308; border-bottom: 2px solid #eab308; padding-bottom: 10px;'>⚙️ SYSTEM INFO</h2>
    """, unsafe_allow_html=True)
    
    # 5. Model Loading (Using Cache so it loads only once)
    @st.cache_resource
    def load_medical_model():
        return tf.keras.models.load_model('chest_xray_5class_densenet.keras')

    try:
        model = load_medical_model()
        st.markdown("""
            <div style='background-color: #0f3a1f; border-left: 4px solid #22c55e; padding: 12px; border-radius: 8px;'>
                <p style='color: #22c55e; margin: 0;'><b>✅ Model Status:</b> READY</p>
                <p style='color: #eab308; margin: 5px 0 0 0; font-size: 0.9em;'>DenseNet-121 • 5 Classes</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown("""
            <div style='background-color: #3a0f0f; border-left: 4px solid #ff4444; padding: 12px; border-radius: 8px;'>
                <p style='color: #ff4444; margin: 0;'><b>❌ Error:</b> Model not found</p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown("---")
    
    st.markdown("""
        <h3 style='color: #eab308;'>📋 CLASS INFORMATION</h3>
    """, unsafe_allow_html=True)
    
    class_info = {
        'Normal': '🟢',
        'COVID-19': '🔴',
        'Bacterial Pneumonia': '🟡',
        'Viral Pneumonia': '🟠',
        'Tuberculosis': '⚫'
    }
    
    for disease, emoji in class_info.items():
        st.markdown(f"<p style='margin: 8px 0; color: #ffffff;'>{emoji} {disease}</p>", unsafe_allow_html=True)

# 6. Correct Class Order (Matches training dataset)
classes = ['Normal', 'COVID-19', 'Bacterial Pneumonia', 'Viral Pneumonia', 'Tuberculosis']

# Define class colors for visualization
class_colors = {
    'Normal': '#22c55e',
    'COVID-19': '#ef4444',
    'Bacterial Pneumonia': '#eab308',
    'Viral Pneumonia': '#f97316',
    'Tuberculosis': '#6366f1'
}

# 7. Main Content Area
st.markdown("""
    <h2 style='color: #eab308; border-bottom: 2px solid #eab308; padding-bottom: 10px;'>📤 IMAGE UPLOAD</h2>
""", unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Drag & Drop your X-Ray Image or Click to Browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
        <div class='card' style='background-color: #0f3a1f; border-left: 4px solid #eab308;'>
            <p style='color: #eab308; font-weight: bold; margin: 0;'>📋 REQUIREMENTS</p>
            <p style='color: #ffffff; font-size: 0.9em; margin: 5px 0 0 0;'>
                ✓ JPG, JPEG, or PNG<br>
                ✓ Size: Any<br>
                ✓ Chest X-Ray Only
            </p>
        </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    st.markdown("---")
    
    # Image Display
    image = Image.open(uploaded_file)
    col_img, col_preview = st.columns([1, 2])
    
    with col_img:
        st.markdown("<h3 style='color: #22c55e;'>📸 Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="X-Ray Scan")
    
    with col_preview:
        st.markdown("<h3 style='color: #22c55e;'>🔍 ANALYSIS READY</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div class='card' style='background-color: #0f3a1f; border-left: 4px solid #22c55e;'>
                <p style='color: #ffffff; margin: 0;'>Click the button below to start AI analysis</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Predict Button
    st.markdown("")
    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("START DIAGNOSIS", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🔄 Processing X-Ray Image..."):
            # --- CRUCIAL PREPROCESSING PIPELINE ---
            img = image.convert('RGB').resize((224, 224))
            img_array = np.array(img)
            
            # DenseNet121 Preprocessing
            img_array = tf.keras.applications.densenet.preprocess_input(img_array)
            
            # Batch Dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # MODEL INFERENCE
            predictions = model.predict(img_array, verbose=0)
            
            # Get Results
            predicted_index = np.argmax(predictions[0])
            predicted_class = classes[predicted_index]
            confidence_score = predictions[0][predicted_index] * 100
        
        st.markdown("---")
        
        # --- RESULTS SECTION ---
        st.markdown("""
            <h2 style='color: #eab308; border-bottom: 2px solid #eab308; padding-bottom: 10px;'>🩺 DIAGNOSIS RESULTS</h2>
        """, unsafe_allow_html=True)
        
        # Main Result Box
        result_color = class_colors.get(predicted_class, '#22c55e')
        st.markdown(f"""
            <div class='result-box' style='border: 3px solid {result_color};'>
                <p style='color: #eab308; margin: 0; font-size: 0.9em;'>PRIMARY DIAGNOSIS</p>
                <h2 style='color: {result_color}; margin: 10px 0; font-size: 2.5em;'>{predicted_class}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence Score
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
                <div class='confidence-box'>
                    <p style='margin: 0; font-size: 0.9em;'>CONFIDENCE SCORE</p>
                    <h1 style='margin: 10px 0; font-size: 3em;'>{confidence_score:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Confidence Assessment
        if confidence_score >= 90:
            assessment = "🟢 VERY HIGH - Results highly reliable"
            color = "#22c55e"
        elif confidence_score >= 75:
            assessment = "🟡 HIGH - Results reliable"
            color = "#eab308"
        elif confidence_score >= 60:
            assessment = "🟠 MODERATE - Recommend review"
            color = "#f97316"
        else:
            assessment = "🔴 LOW - Recommend additional testing"
            color = "#ef4444"
        
        st.markdown(f"""
            <div class='card' style='border-left: 4px solid {color};'>
                <p style='color: {color}; margin: 0; font-weight: bold;'>{assessment}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Breakdown
        st.markdown("""
            <h3 style='color: #eab308;'> PROBABILITY BREAKDOWN</h3>
        """, unsafe_allow_html=True)
        
        # Create columns for probability display
        prob_cols = st.columns(len(classes))
        
        for idx, class_name in enumerate(classes):
            prob = predictions[0][idx] * 100
            is_predicted = (idx == predicted_index)
            
            with prob_cols[idx]:
                box_color = class_colors.get(class_name, '#22c55e')
                border_style = f"3px solid {box_color}" if is_predicted else f"1px solid {box_color}"
                background = f"rgba(34, 197, 94, 0.1)" if is_predicted else "#1a1a1a"
                
                st.markdown(f"""
                    <div class='prob-container' style='border: {border_style}; background-color: {background};'>
                        <p style='color: {box_color}; margin: 0; font-weight: bold;'>{class_name}</p>
                        <p style='color: #eab308; margin: 8px 0 0 0; font-size: 1.3em; font-weight: bold;'>{prob:.1f}%</p>
                        <div style='background-color: #0a0a0a; border-radius: 8px; height: 8px; margin-top: 8px; overflow: hidden;'>
                            <div style='background: linear-gradient(to right, {box_color}, #eab308); width: {prob}%; height: 100%;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Disclaimer
        st.markdown("""
            <div class='card' style='border-left: 4px solid #f97316; background-color: #1a0f0f;'>
                <p style='color: #f97316; margin: 0; font-weight: bold;'>⚠️ DISCLAIMER</p>
                <p style='color: #ffffff; margin: 10px 0 0 0; font-size: 0.9em;'>
                    This AI system is designed to assist medical professionals and should not be used as a standalone diagnostic tool. 
                    Always consult with qualified healthcare providers for medical diagnosis and treatment decisions.
                </p>
            </div>
        """, unsafe_allow_html=True)
else:
    # Empty State
    st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h2 style='color: #eab308; margin: 0;'>👋 WELCOME TO X-RAY ANALYZER</h2>
            <p style='color: #22c55e; margin: 20px 0; font-size: 1.2em;'>Upload a chest X-ray to get started</p>
            <div class='card' style='background-color: #0f3a1f; border-left: 4px solid #22c55e; margin: 30px auto; max-width: 400px;'>
                <p style='color: #ffffff; margin: 0;'>
                    🩻 Supported Formats: JPG, JPEG, PNG<br>
                     Model: DenseNet-121<br>
                     Classes: 5 Disease Types<br>
                     Processing: ~2-3 seconds
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)