import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
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
    *PharmD, MSc (Natural Drug Design & Discovery)* *Pharmacist, Lecturer & Aspirant Researcher*
    
    **Methodology:**
    - **Model:** Random Forest Classifier  
    - **Imputation:** MICE (IterativeImputer)
    - **Interpretability:** SHAP (XAI)
    - **Safety Layer:** Hard Clinical Rules
    - **Data:** UCI Heart Disease Dataset
    """)
    st.warning("**Disclaimer:** This is a research prototype for educational purposes.")

# ==========================================
# 2. LOAD THE SAVED ARTIFACTS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # Load the main pipeline
        model = joblib.load('heart_disease_pipeline.joblib')
        
        # Load Imputer 
        try:
            imputer = joblib.load('mice_imputer(1)')
        except:
            imputer = joblib.load('mice_imputer(1).joblib')
            
        # Load Columns 
        try:
            cols = joblib.load('model_columns1(2)')
        except:
            cols = joblib.load('model_columns1(2).joblib')
            
        return model, imputer, cols
    except Exception as e:
        st.error(f"Critical Error Loading Files: {e}")
        st.stop()

# Initialize assets
model, mice_imputer, model_cols = load_assets()

# ==========================================
# 3. CLINICAL DATA ENTRY 
# ==========================================
st.title("🫀 Cardiovascular Risk Prediction Prototype")

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

    submit_button = st.form_submit_button("Analyze Risk")

# ==========================================
# 4. PREDICTION LOGIC WITH CLINICAL RULES
# ==========================================
if submit_button:
    # --- A. CLINICAL RULE LAYER ---
    clinical_alert = False
    reasons = []

    # Rule 1: ECG Abnormality
    if restecg == "st-t wave abnormality":
        clinical_alert = True
        reasons.append("ST-T Wave Abnormality detected")
        
    # Rule 2: Age above 50
    if age > 50:
        clinical_alert = True
        reasons.append(f"Age Risk Factor ({age} yrs)")

    # Rule 3: Cholesterol above 200
    if chol > 200:
        clinical_alert = True
        reasons.append(f"Elevated Cholesterol ({chol} mg/dl)")

    # Rule 4: Typical Angina
    if cp == "typical angina":
        clinical_alert = True
        reasons.append("Typical Angina reported")

    # --- B. MACHINE LEARNING LAYER ---
    # Create dataframe (formatting exact strings/numbers for the model)
    input_df = pd.DataFrame([{
        'age': age, 
        'sex': sex, 
        'cp': cp, 
        'trestbps': trestbps,
        'chol': chol, 
        'fbs': 1 if fbs else 0,  
        'restecg': restecg,      
        'thalch': thalch, 
        'exang': 1 if exang else 0, 
        'oldpeak': oldpeak
    }])

    # One-Hot Encoding & Reindexing to match training data structure exactly
    input_encoded = pd.get_dummies(input_df)
    final_input = input_encoded.reindex(columns=model_cols, fill_value=0)

    try:
        # Impute & Predict probabilities
        final_input_imputed = mice_imputer.transform(final_input)
        prob = model.predict_proba(final_input_imputed)[0][1]

        # --- C. HYBRID DECISION OUTPUT ---
        st.divider()
        
        # Trigger High Risk if EITHER clinical rules are met OR model is > 50% sure
        if clinical_alert or prob > 0.5:
            st.error("### 🚨 HIGH RISK DETECTED")
            
            # Show the model's baseline calculation
            st.write(f"**Base Model Probability:** {prob:.2%}")
            
            # Highlight the hard clinical rules that triggered the alert
            if clinical_alert:
                st.warning(f"**Mandatory Clinical Flags:** {', '.join(reasons)}")
                st.info("Medical recommendation: Further clinical investigation required based on the above red flags.")
        else:
            st.success(f"### ✅ LOW RISK DETECTED (Probability: {prob:.2%})")
            st.info("**Clinical Note:** Patient parameters and inputs fall within standard baselines.")

        # --- D. SHAP EXPLAINER ---
        st.divider()
        st.subheader("🔍 Feature Importance (SHAP)")
        
        # Use TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_input_imputed)
        
        fig, ax = plt.subplots()
        # [1] maps to the 'High Risk' class in binary classification
        shap.summary_plot(shap_values[1], final_input, plot_type="bar", show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Computation Error: {e}")
