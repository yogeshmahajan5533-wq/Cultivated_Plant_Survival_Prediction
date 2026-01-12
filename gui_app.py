import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Language Configuration
# -----------------------------
if 'language' not in st.session_state:
    st.session_state.language = 'english'

def toggle_language():
    if st.session_state.language == 'english':
        st.session_state.language = 'hindi'
    else:
        st.session_state.language = 'english'

# -----------------------------
# Language Content
# -----------------------------
content = {
    'english': {
        'title': "🌱 Plant Survival Prediction System",
        'description': "This system predicts plant survival suitability using crop-specific One-Class SVM models.",
        'select_plant': "🌿 Select Plant / Crop",
        'enter_params': "🌡️ Enter Environmental Parameters",
        'predict_button': "🔍 Predict Survival",
        'result_title': "📊 Prediction Result",
        'selected_plant': "🌱 **Selected Plant:**",
        'suitable': "✅ Conditions are SUITABLE for this plant",
        'risky': "⚠️ Conditions are RISKY for this plant",
        'survival_prob': "🌿 Survival Probability (%)",
        'feature_descriptions': {
            'N': "**Nitrogen (N):** Essential for leaf growth and green color. Affects protein and chlorophyll formation.",
            'P': "**Phosphorus (P):** Important for root development, flowering, and fruiting. Helps in energy transfer.",
            'K': "**Potassium (K):** Improves disease resistance and water regulation. Essential for overall plant health.",
            'temperature': "**Temperature:** Daily average temperature in °C. Affects germination, growth, and yield.",
            'humidity': "**Humidity:** Relative humidity in %. Affects transpiration and disease occurrence.",
            'ph': "**Soil pH:** Acidity/alkalinity level. Affects nutrient availability to plants.",
            'rainfall': "**Rainfall:** Annual rainfall in mm. Determines irrigation needs and water availability."
        },
        'site_description': """
        ### 📍 About This System
        This AI-powered system helps Indian farmers determine whether their local conditions are suitable for specific crops. 
        The model analyzes 7 key environmental parameters and provides a survival probability percentage.
        
        **Features:**
        • **Crop-Specific Models:** Trained on Indian agricultural data
        • **Real-time Prediction:** Immediate results based on your inputs
        • **Scientific Accuracy:** Uses One-Class SVM machine learning
        • **User-Friendly:** Designed specifically for farmers
        
        **Note:** This tool provides guidance based on statistical analysis. Local soil conditions, 
        farming practices, and microclimates may affect actual results.
        """,
        'language_button': "हिंदी में देखें / View in Hindi"
    },
    'hindi': {
        'title': "🌱 पौधा जीवित रहने की भविष्यवाणी प्रणाली",
        'description': "यह प्रणाली फसल-विशिष्ट वन-क्लास एसवीएम मॉडल का उपयोग करके पौधे के जीवित रहने की उपयुक्तता का अनुमान लगाती है।",
        'select_plant': "🌿 फसल / पौधा चुनें",
        'enter_params': "🌡️ पर्यावरणीय मापदंड दर्ज करें",
        'predict_button': "🔍 भविष्यवाणी करें",
        'result_title': "📊 भविष्यवाणी परिणाम",
        'selected_plant': "🌱 **चुना गया पौधा:**",
        'suitable': "✅ परिस्थितियाँ इस फसल के लिए उपयुक्त हैं",
        'risky': "⚠️ परिस्थितियाँ इस फसल के लिए जोखिम भरी हैं",
        'survival_prob': "🌿 जीवित रहने की संभावना (%)",
        'feature_descriptions': {
            'N': "**नाइट्रोजन (N):** पत्तियों के विकास और हरे रंग के लिए आवश्यक। प्रोटीन और क्लोरोफिल निर्माण को प्रभावित करता है।",
            'P': "**फॉस्फोरस (P):** जड़ विकास, फूल आना और फलने के लिए महत्वपूर्ण। ऊर्जा हस्तांतरण में मदद करता है।",
            'K': "**पोटेशियम (K):** रोग प्रतिरोधक क्षमता और जल विनियमन में सुधार करता है। समग्र पौध स्वास्थ्य के लिए आवश्यक।",
            'temperature': "**तापमान:** डिग्री सेल्सियस में दैनिक औसत तापमान। अंकुरण, वृद्धि और उपज को प्रभावित करता है।",
            'humidity': "**आर्द्रता:** प्रतिशत में सापेक्ष आर्द्रता। वाष्पोत्सर्जन और रोग घटना को प्रभावित करती है।",
            'ph': "**मृदा pH:** अम्लीयता/क्षारीयता स्तर। पौधों को पोषक तत्वों की उपलब्धता को प्रभावित करता है।",
            'rainfall': "**वर्षा:** मिलीमीटर में वार्षिक वर्षा। सिंचाई की आवश्यकताएं और जल उपलब्धता निर्धारित करती है।"
        },
        'site_description': """
        ### 📍 इस प्रणाली के बारे में
        यह एआई-संचालित प्रणाली भारतीय किसानों को यह निर्धारित करने में मदद करती है कि उनकी स्थानीय परिस्थितियाँ विशिष्ट फसलों के लिए उपयुक्त हैं या नहीं।
        यह मॉडल 7 प्रमुख पर्यावरणीय मापदंडों का विश्लेषण करता है और जीवित रहने की संभावना प्रतिशत प्रदान करता है।
        
        **विशेषताएँ:**
        • **फसल-विशिष्ट मॉडल:** भारतीय कृषि डेटा पर प्रशिक्षित
        • **रीयल-टाइम भविष्यवाणी:** आपके इनपुट के आधार पर तत्काल परिणाम
        • **वैज्ञानिक सटीकता:** वन-क्लास एसवीएम मशीन लर्निंग का उपयोग करता है
        • **उपयोगकर्ता के अनुकूल:** विशेष रूप से किसानों के लिए डिज़ाइन किया गया
        
        **नोट:** यह उपकरण सांख्यिकीय विश्लेषण के आधार पर मार्गदर्शन प्रदान करता है। स्थानीय मृदा स्थितियाँ,
        कृषि पद्धतियाँ और सूक्ष्म जलवायु वास्तविक परिणामों को प्रभावित कर सकती हैं।
        """,
        'language_button': "View in English / अंग्रेजी में देखें"
    }
}

# Get current language content
lang = st.session_state.language
text = content[lang]

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title=text['title'],
    layout="centered"
)

# Language toggle button at top
col1, col2, col3 = st.columns([3, 1, 1])
with col3:
    if st.button(text['language_button']):
        toggle_language()
        st.rerun()

st.title(text['title'])
st.write(text['description'])

# -----------------------------
# Feature list (must match training order)
# -----------------------------
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# -----------------------------
# GUI input ranges (derived from dataset)
# -----------------------------
gui_ranges = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 210),
    "temperature": (5, 45),
    "humidity": (10, 100),
    "ph": (3.0, 10.0),
    "rainfall": (20, 300)
}

# -----------------------------
# Load available plant models
# -----------------------------
MODEL_DIR = "trained_models"

plant_names = sorted([
    f.replace("_ocsvm.pkl", "")
    for f in os.listdir(MODEL_DIR)
    if f.endswith("_ocsvm.pkl")
])

# -----------------------------
# Plant / Crop Name dropdown
# -----------------------------
st.subheader(text['select_plant'])

plant = st.selectbox(
    "Plant / Crop Name" if lang == 'english' else "फसल / पौधा का नाम",
    plant_names
)

# -----------------------------
# Input section with detailed descriptions
# -----------------------------
st.subheader(text['enter_params'])

input_values = []

for feature in features:
    min_val, max_val = gui_ranges[feature]
    
    # Feature description in expander
    with st.expander(f"{feature.capitalize()} - {text['feature_descriptions'][feature].split('**')[1].split('**')[0]}"):
        st.markdown(text['feature_descriptions'][feature])
    
    value = st.slider(
        label=feature.capitalize(),
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),
        step=0.1,
        help=text['feature_descriptions'][feature]
    )
    input_values.append(value)

X_input = np.array(input_values).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
if st.button(text['predict_button']):
    try:
        # Load model and scaler
        scaler = joblib.load(f"{MODEL_DIR}/{plant}_scaler.pkl")
        model = joblib.load(f"{MODEL_DIR}/{plant}_ocsvm.pkl")

        # Scale input
        X_scaled = scaler.transform(X_input)

        # Model prediction
        prediction = model.predict(X_scaled)[0]
        decision_score = model.decision_function(X_scaled)[0]

        # Normalize decision score to survival %
        survival_percent = 100 / (1 + np.exp(-decision_score))

        # -----------------------------
        # Output section
        # -----------------------------
        st.subheader(text['result_title'])

        st.write(f"{text['selected_plant']} {plant.capitalize()}")

        if prediction == 1:
            st.success(text['suitable'])
        else:
            st.warning(text['risky'])

        st.metric(
            label=text['survival_prob'],
            value=f"{survival_percent:.2f}"
        )

        # Additional interpretation
        st.info("💡 **Interpretation:** " + 
               ("Higher percentage indicates better suitability for the selected crop. " if lang == 'english' else "उच्च प्रतिशत चुनी गई फसल के लिए बेहतर उपयुक्तता को दर्शाता है। ") +
               ("Values above 50% generally indicate favorable conditions." if lang == 'english' else "50% से अधिक मान आम तौर पर अनुकूल परिस्थितियों को दर्शाते हैं।"))

    except Exception as e:
        st.error(f"Error occurred: {e}")

# -----------------------------
# Site description at bottom
# -----------------------------
st.markdown("---")
st.markdown(text['site_description'])

# Add footer
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.caption("🌾 Made for Indian Farmers | किसानों के लिए बनाया गया")
with footer_col2:
    if st.button("🔄 " + ("Switch Language" if lang == 'english' else "भाषा बदलें"), type="secondary"):
        toggle_language()
        st.rerun()
