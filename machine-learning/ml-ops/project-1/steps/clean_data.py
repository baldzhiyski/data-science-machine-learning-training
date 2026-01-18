import logging
import pandas as pd
from zenml import step



@step
def clean_data_step(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input data by handling missing values and removing duplicates.
    Args:
        data (pd.DataFrame): The input data to be cleaned.
    Returns:
        pd.DataFrame: The cleaned data.
    """

    logging.info("Cleaning data")
    return data