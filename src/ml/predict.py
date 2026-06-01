import joblib
import shap
import pandas as pd
import numpy as np
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
    
    # Safely extract a 1D array regardless of the SHAP version format
    if isinstance(shap_values, list):
        impacts = shap_values[1][0]
    else:
        shap_array = np.array(shap_values)
        if len(shap_array.shape) == 3:
            impacts = shap_array[0, :, 1]
        else:
            impacts = shap_array[0]
            
    # Ultimate safety net: force flatten to 1D
    impacts = np.array(impacts).flatten()
    
    # Extract feature importance for this specific prediction
    feature_names = input_df.columns
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': impacts
    }).sort_values(by='Impact', key=abs, ascending=False)
    
    return proba, band, shap_df