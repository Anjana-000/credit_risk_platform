import joblib
import shap
import pandas as pd
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict_risk(input_df):
    """Predicts default probability and returns SHAP values."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")
        
    pipeline = joblib.load(MODEL_PATH)
    
    # Ensure inputs match training columns
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    proba = pipeline.predict_proba(input_df)[:, 1][0]
    
    # Business Decision Rules
    if proba < 0.30:
        band = "Low Risk"
    elif proba < 0.60:
        band = "Medium Risk"
    else:
        band = "High Risk"
        
    # Explainable AI (SHAP)
    processed_input = preprocessor.transform(input_df)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(processed_input)
    
    # Extract feature importance for this specific prediction
    feature_names = input_df.columns
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    }).sort_values(by='Impact', key=abs, ascending=False)
    
    return proba, band, shap_df