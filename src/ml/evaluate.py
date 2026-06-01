from sklearn.metrics import roc_auc_score, classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(pipeline, X_test, y_test):
    logger.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    return roc_auc, report