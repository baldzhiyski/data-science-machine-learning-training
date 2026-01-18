import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract base class for data cleaning strategies.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Concrete strategy for preprocessing data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data preprocessing...")

        try:
            # Some simple preprocessing
            data = data.drop_duplicates()
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            logging.info("Data preprocessing completed.")
            return data
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

class DataDevideStrategy(DataStrategy):
    """
    Concrete strategy for dividing data into features and target.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        logging.info("Dividing data into features and target variable...")

        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error during data division: {e}")
            raise

class DataCleaning:
    """
    Context class for data cleaning using different strategies.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self._data = data
        self._strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self._strategy.handle_data(self._data)
