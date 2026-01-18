import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path):
        self.data_path = data_path
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_data(self):
        """ Ingest data from a CSV file."""
        self.logger.info(f"Reading data from {self.data_path}")
        data = pd.read_csv(self.data_path)
        self.logger.info(f"Data shape: {data.shape}")
        return data

@step
def ingest_data_step(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a CSV file located at data_path.
    Args:
        data_path (str): The path to the CSV file to ingest.
    Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        ingestor = IngestData(data_path)
        data = ingestor.get_data()
        return data
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise