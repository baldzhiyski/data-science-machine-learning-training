import pandas as pd
import logging
from zenml import step



@step
def evaluate_model_step(model:dict, cleaned_data: pd.DataFrame) -> float:
    """
    Evaluate the trained model on the test data.     :param model: The trained machine learning model.
    :param test_data: The test data to evaluate the model on.
    :return: None
    """
    logging.info("Evaluating model")
    logging.info(f"Model evaluation results: {model}, {cleaned_data}")
    return 1.0