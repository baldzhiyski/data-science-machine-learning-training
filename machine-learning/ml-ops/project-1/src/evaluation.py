import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import numpy as np

class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def calculate_score(self, y_true, y_pred):
        """
        Calculate the evaluation score.

        Parameters:
        y_true : array-like, shape (n_samples,)
            True target values.
        y_pred : array-like, shape (n_samples,)
            Predicted target values.
        """
        pass

class RMSE(Evaluation):
    """
    Root Mean Squared Error (RMSE) evaluation metric.
    """

    def calculate_score(self, y_true, y_pred):
        """
        Calculate the RMSE score.

        Parameters:
        y_true : array-like, shape (n_samples,)
            True target values.
        y_pred : array-like, shape (n_samples,)
            Predicted target values.

        Returns:
        float
            The RMSE score.
        """


        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        logging.info(f"Calculated RMSE: {rmse}")
        return rmse

class R2(Evaluation):

    """
    R-squared (R2) evaluation metric.
    """

    def calculate_score(self, y_true, y_pred):
        """
        Calculate the R2 score.

        Parameters:
        y_true : array-like, shape (n_samples,)
            True target values.
        y_pred : array-like, shape (n_samples,)
            Predicted target values.

        Returns:
        float
            The R2 score.
        """

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        logging.info(f"Calculated R2: {r2}")
        return r2

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("Entered the calculate_score method of the MSE class")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("The mean squared error value is: " + str(mse))
            return mse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MSE class. Exception message:  "
                + str(e)
            )
            raise e