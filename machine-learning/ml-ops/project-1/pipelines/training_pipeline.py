from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data_step
from steps.clean_data import clean_data_step
from steps.evaluate_model import  evaluate_model_step
from steps.train_model import  train_model_step
from steps.config import ModelNameConfig

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    """A training pipeline that ingests data, preprocesses it ,
    trains a model, and evaluates the model."""
    data = ingest_data_step(data_path=data_path)
    X_train , X_test , y_train , y_test = clean_data_step(data=data)

    config = ModelNameConfig(model_name="linear_regression", fine_tuning=False)

    model = train_model_step(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, config=config)
    r2_score , mse = evaluate_model_step(model=model, X_test=X_test, y_test=y_test)
