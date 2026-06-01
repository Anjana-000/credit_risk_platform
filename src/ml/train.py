import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from src.data.loader import load_csv
from src.data.preprocessor import clean_data, build_preprocessing_pipeline
from src.utils.config import MODEL_PATH, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train():
    logger.info("Starting model training pipeline...")
    df = load_csv("application_train.csv", nrows=50000) # Subset for performance
    X, y = clean_data(df, TARGET_COL)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = build_preprocessing_pipeline()
    
    # Handle Class Imbalance using class_weight='balanced'
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    logger.info("Fitting the model...")
    pipeline.fit(X_train, y_train)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return pipeline, X_test, y_test

if __name__ == "__main__":
    train()