import logging
import pandas as pd
from zenml import step


@step
def train_model_step(cleaned_data: pd.DataFrame) -> dict:
    """
    Train a machine learning model on the cleaned data.
    Args:
        cleaned_data (pd.DataFrame): The cleaned input data.
    Returns:
        None
    """
    logging.info("Training model")
    return {"model": "dummy", "rows": int(cleaned_data.shape[0])}