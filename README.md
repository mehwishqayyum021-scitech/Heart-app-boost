# Heart-app-boost
Heart risk predictor tool
# Cardiovascular Risk Assessment tool

This project implements a clinical decision support tool for heart disease prediction. It transitions from traditional diagnostic methods to **AI implementation science** by using robust machine learning pipelines that handle real-world "messy" data (missing values) while providing human-interpretable explanations.

#🛠️ Tech Stack & Dependencies
This project is built using the following core libraries:
* **Pandas & NumPy:** For data manipulation and numerical processing.
* **Scikit-Learn:** For the `IterativeImputer` (MICE), `Pipeline`, and `K-Fold` cross-validation logic.
* **XGBoost:** The primary gradient-boosting classifier for high-performance prediction.
* **SHAP:** For Explainable AI (XAI) to interpret individual clinical risk factors.
* **Streamlit:** To provide a real-time web interface for clinicians.
* **Joblib:** For efficient model serialization.

#Repository Structure
To keep the project lightweight, we use a single Pipeline object that contains both the imputer and the model.

1.  **`heart_disease_predictor.ipynb`**: The development notebook containing the research logic and K-Fold validation.
2.  **`heart_app.py`**: The Streamlit web application code.
3.  **`heart_disease_pipeline.joblib`**: The serialized **XGBoost Pipeline** (includes both the MICE imputer and the trained model).
4.  **`model_columns1.joblib`**: A helper file ensuring the input data matches the training feature order.

## Key Features
* **MICE Imputation:** Automatically handles missing patient data during both training and real-time inference.
* **5-Fold Cross-Validation:** Ensures the model's accuracy is stable and not a result of overfitting.
* **Clinical Interpretability:** Every prediction includes a SHAP force plot, showing exactly which patient vitals (Age, BP, Cholesterol) increased or decreased the risk score.
* "This prototype utilizes a Hybrid Decision Engine. It combines a Random Forest probabilistic model with a Deterministic Clinical Layer to ensure that high-risk markers (like ECG abnormalities) are never missed by the algorithm.

## How to Run
1. Install requirements:
   `pip install pandas numpy scikit-learn xgboost shap streamlit joblib`
2. Run the app:
   `streamlit run heart_app.py`

