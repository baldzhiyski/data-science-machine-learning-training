from typing import Tuple

import pandas as pd
import logging
from zenml import step
import mlflow
from src.evaluation import RMSE, R2, MSE
from typing_extensions import Annotated
from sklearn.base import RegressorMixin

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker)
def evaluate_model_step(model:RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[
float, "R2 Score"], Annotated[float, "MSE Score"]]:
    """
    Evaluate the trained model on the test data.
    :param model: The trained machine learning model.
    :param X_test: The test features.
    :param y_test: The true target values for the test set.
    :return: The evaluation score.
    """
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)

    rmse = RMSE().calculate_score(y_test, y_pred)
    mlflow.log_metric("RMSE", rmse)

    r2 = R2().calculate_score(y_test, y_pred)
    mlflow.log_metric("R2", r2)

    mse = MSE().calculate_score(y_test, y_pred)
    mlflow.log_metric("MSE", mse)

    logging.info(f"Model evaluation results: RMSE={rmse}, R2={r2}, MSE={mse}")

    return r2,mse