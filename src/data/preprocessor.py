import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_data(df, target_col='TARGET'):
    """Handles missing values and selects numeric features."""
    logger.info("Cleaning data and dropping missing columns > 50%")
    
    # Drop columns with more than 50% missing values
    limit_per_column = len(df) * 0.50
    df = df.dropna(thresh=limit_per_column, axis=1)
    
    # For MVP, restrict to numeric features to build a robust baseline model
    numeric_df = df.select_dtypes(include=[np.number])
    
    X = numeric_df.drop(columns=[target_col, 'SK_ID_CURR'], errors='ignore')
    y = numeric_df[target_col] if target_col in numeric_df.columns else None
    
    return X, y

def build_preprocessing_pipeline():
    """Returns a scikit-learn pipeline"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])