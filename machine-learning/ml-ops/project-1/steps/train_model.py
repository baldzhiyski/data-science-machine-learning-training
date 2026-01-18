import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from config import  ModelNameConfig

@step
def train_model_step(X_train, X_test, y_train, y_test, config: ModelNameConfig) -> RegressorMixin:
    """
    Train a machine learning model on the cleaned data.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
    Returns:
           RegressorMixin: The trained model.
    """
    logging.info("Training model")
    model = None

    if config.model_name == "linear_regression":
        model_instance = LinearRegressionModel()
        model = model_instance.train(X_train, y_train)
        logging.info("Linear Regression model trained.")



