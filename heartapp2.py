import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. PAGE SETUP & RESEARCHER CREDITS
# ==========================================
st.set_page_config(page_title="Heart Risk CDSS", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #e1e8f0 100%);
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
    st.warning("**Disclaimer:** This is a research prototype for educational purposes.")

# ==========================================
# 2. LOAD THE SAVED ARTIFACTS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # Using the exact names from your repository list
        model = joblib.load('heart_disease_pipeline.joblib')
        imputer = joblib.load('mice_imputer(1)') 
        cols = joblib.load('model_columns1(2)')
        return model, imputer, cols
    except Exception as e:
        st.error(f" Error loading files: {e}")
        st.info("Check if filenames in repo include .joblib or .pkl extensions.")
        st.stop()

# Initialize assets
model, mice_imputer, model_cols = load_assets()

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

    # FIX: Use st.form_submit_button instead of st.button
    submit_button = st.form_submit_button("Analyze Risk")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if submit_button:
    # 1. Create initial dataframe
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak
    }])

    # 2. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)

    # 3. Reindex using the loaded columns (model_cols)
    # This ensures the features match the training set exactly
    final_input = input_encoded.reindex(columns=model_cols, fill_value=0)

    # 4. Apply MICE imputation
    try:
        final_input_imputed = mice_imputer.transform(final_input)
        
        # 5. Predict
        prediction = model.predict(final_input_imputed)
        probability = model.predict_proba(final_input_imputed)[0][1]

        # 6. Display Results
        st.divider()
        if prediction[0] == 1:
            st.error(f"### High Risk Detected (Probability: {probability:.2%})")
            st.info("**Clinical Note:** Immediate cardiovascular consultation recommended.")
        else:
            st.success(f"### Low Risk Detected (Probability: {probability:.2%})")
            st.info("**Clinical Note:** Patient parameters within standard baseline.")
            
    except Exception as e:
        st.error(f"Processing Error: {e}")
