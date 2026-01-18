from typing import Tuple

import pandas as pd
import logging
from zenml import step
from src.evaluation import RMSE, R2, MSE
from typing_extensions import Annotated
from sklearn.base import RegressorMixin


@step
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
    r2 = R2().calculate_score(y_test, y_pred)
    mse = MSE().calculate_score(y_test, y_pred)
    logging.info(f"Model evaluation results: RMSE={rmse}, R2={r2}, MSE={mse}")

    return r2,mse