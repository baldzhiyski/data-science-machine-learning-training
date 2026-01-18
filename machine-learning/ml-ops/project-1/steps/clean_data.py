import logging
from typing import Tuple

import pandas as pd
from sqlalchemy.sql.annotation import Annotated
from zenml import step
from src.data_cleaning import DataPreprocessStrategy,DataCleaning,DataDevideStrategy



@step
def clean_data_step(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clean the input data by handling missing values and removing duplicates.
    Args:
        data (pd.DataFrame): The input data to be cleaned.
    Returns:
        pd.DataFrame: The cleaned data.
    """

    try:
        logging.info("Cleaning data...")
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data,preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        devide_strategy = DataDevideStrategy()
        data_cleaning = DataCleaning(processed_data,devide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data cleaned and devided ...")

        return X_train,X_test,y_train,y_test
    except Exception:
        logging.error("An error occurred while cleaning the data.")
        raise