import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE SETUP & RESEARCHER CREDITS
# ==========================================
st.set_page_config(page_title="Heart Risk CDSS", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #e1e8f0 100%);
    }
    .stButton>button {
        background-color: #003366; 
        color: #e1e8f0;
        font-weight: bold;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🔬 Clinical Metadata")
    st.info("""
    **Researcher:** MQ  
    *PharmD, MSc (Natural Drug Design & Discovery)* *Pharmacist & Aspirant Researcher*
    
    **Methodology:**
    - **Model:** Random Forest Classifier  
    - **Imputation:** MICE (IterativeImputer)
    - **Interpretability:** SHAP (XAI)
    - **Data:** UCI Heart Disease Dataset
    """)
    st.warning("**Disclaimer:** This is a research prototype for educational purposes. It is not a validated clinical diagnostic tool.")

# ==========================================
# 2. LOAD THE SAVED ARTIFACTS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('heart_model1.joblib')
    imputer = joblib.load('mice_imputer.joblib')
    cols = joblib.load('model_columns1.joblib')
    return model, imputer, cols

try:
    rf_model, mice_imputer, saved_columns = load_assets()
except Exception as e:
    st.error("Model files not found. Ensure .joblib files are in the repository.")
    st.stop()

# ==========================================
# 3. CLINICAL DATA ENTRY 
# ==========================================
st.title("🫀 Cardiovascular Risk Prediction Prototype")
st.write("Enter patient parameters below to generate a risk probability and clinical audit.")

with st.form("clinical_input_form"):
    col1, col2 = st.columns(2)


with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting BP (mm Hg)", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120", [True, False])
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t wave abnormality"])
    thalch = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Induced Angina", [True, False])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 0.0)

# 3. Prediction Logic
if st.button("Analyze Risk"):
    # Create initial dataframe from inputs
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak
    }])

    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)

    # REINDEX: FIXED the variable name here to match the loaded 'model_columns'
    final_input = input_encoded.reindex(columns=model_columns, fill_value=np.nan)

    # Ensure float datatype
    final_input = final_input.astype(float)

    # Apply MICE imputation
    final_input_imputed = mice_imputer.transform(final_input)

    # Predict
    prediction = model.predict(final_input_imputed)
    probability = model.predict_proba(final_input_imputed)[0][1]

    # Display Results
    st.divider()
    if prediction[0] == 1:
        st.error(f"High Risk Detected (Probability: {probability:.2%})")
        st.info("Medical recommendation: Further clinical investigation required.")
    else:
        st.success(f"Low Risk Detected (Probability: {probability:.2%})")
