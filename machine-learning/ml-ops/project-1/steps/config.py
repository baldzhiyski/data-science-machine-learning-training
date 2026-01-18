from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "linear_regression"
    fine_tuning: bool = False
