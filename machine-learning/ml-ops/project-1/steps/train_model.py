import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import  ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model_step(X_train, X_test, y_train, y_test, config: ModelNameConfig) -> RegressorMixin:
    logging.info("Training model")

    if config.model_name != "linear_regression":
        raise ValueError(f"Model {config.model_name} is not supported.")

    model_instance = LinearRegressionModel()


    sklearn_model: RegressorMixin = model_instance.train(X_train, y_train)


    mlflow.sklearn.log_model(
        sk_model=sklearn_model,
        name="model",
        registered_model_name="model",
    )

    logging.info("Linear Regression model trained and logged to MLflow.")
    return sklearn_model