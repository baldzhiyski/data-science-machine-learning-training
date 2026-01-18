import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the provided data.

        Parameters:
        X_train : array-like, shape (n_samples, n_features)
            Training data.
        y_train : array-like, shape (n_samples,)
            Target values.
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model implementation.
    """

    def __init__(self):
        self.coefficients = None
        logging.info("LinearRegressionModel instance created.")

    def train(self, X_train, y_train,**kwargs):
        """
        Train the Linear Regression model using the Normal Equation.

        Parameters:
        X_train : array-like, shape (n_samples, n_features)
            Training data.
        y_train : array-like, shape (n_samples,)
            Target values.
        """

        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        self.coefficients = model.coef_
        logging.info("Model trained successfully with coefficients: %s", self.coefficients)
        return model
