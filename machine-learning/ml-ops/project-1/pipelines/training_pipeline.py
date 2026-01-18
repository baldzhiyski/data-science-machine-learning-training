from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data_step
from steps.clean_data import clean_data_step
from steps.evaluate_model import  evaluate_model_step
from steps.train_model import  train_model_step

@pipeline
def training_pipeline(data_path: str):
    """A training pipeline that ingests data, preprocesses it ,
    trains a model, and evaluates the model."""
    data = ingest_data_step(data_path=data_path)
    cleaned = clean_data_step(data=data)
    model = train_model_step(cleaned_data=cleaned)
    evaluate_model_step(model=model, cleaned_data=cleaned)
