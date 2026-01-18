# pipelines/deployment_pipeline.py
import json
import subprocess
import time
import socket
from typing import List

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel
from zenml import pipeline, step

from steps.clean_data import clean_data_step
from steps.evaluate_model import evaluate_model_step
from steps.ingest_data import ingest_data_step
from steps.train_model import train_model_step
from steps.config import ModelNameConfig
from steps.utils import get_data_for_test


# ----------------------------
# Helpers
# ----------------------------
FEATURE_COLUMNS: List[str] = [
    "payment_sequential",
    "payment_installments",
    "payment_value",
    "price",
    "freight_value",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]


def json_payload_to_dataframe(payload: str) -> pd.DataFrame:
    """Convert stored JSON into the DataFrame MLflow serving expects."""
    data = json.loads(payload)

    rows = data.get("data", [])
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    # ensure numeric types (optional but helps avoid serving errors)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_server(base_url: str, timeout_s: int) -> bool:
    """Wait until MLflow serving responds."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            # MLflow serving usually returns 404 on /
            r = requests.get(base_url, timeout=0.5)
            if r.status_code in (200, 404):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ----------------------------
# Steps
# ----------------------------
@step(enable_cache=False)
def dynamic_importer() -> str:
    return get_data_for_test()


class DeploymentTriggerConfig(BaseModel):
    max_mse: float = 1.0  # lower is better


@step(enable_cache=False)
def deployment_trigger(mse: float, config: DeploymentTriggerConfig) -> bool:
    return mse <= config.max_mse


class ServeConfig(BaseModel):
    model_name: str = "model"
    model_stage_or_version: str = "latest"  # "latest" or "1" or "Production"
    host: str = "127.0.0.1"
    port: int = 5001
    timeout_s: int = 60


@step(enable_cache=False)
def start_mlflow_model_server(config: ServeConfig) -> str:
    """
    Starts MLflow model serving locally (Windows-safe) and returns base URL.

    Uses:
      mlflow models serve -m models:/<name>/<stage_or_version> -h <host> -p <port> --no-conda
    """

    base_url = f"http://{config.host}:{config.port}"

    # If already running, just return
    if port_open(config.host, config.port) and wait_for_server(base_url, timeout_s=3):
        return base_url

    model_uri = f"models:/{config.model_name}/{config.model_stage_or_version}"

    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-h",
        config.host,
        "-p",
        str(config.port),
        "--no-conda",
    ]

    # Log to file for debugging (VERY important on Windows)
    log_path = "mlflow_serve.log"
    log_file = open(log_path, "w", encoding="utf-8")

    # IMPORTANT:
    # - shell=False with list is correct
    # - start_new_session=True helps detach on Windows
    subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        shell=False,
        start_new_session=True,
    )

    # Wait until server is reachable
    if wait_for_server(base_url, timeout_s=config.timeout_s):
        return base_url

    raise RuntimeError(
        f"MLflow model server did not start at {base_url}. "
        f"Check {log_path} for the reason."
    )


@step(enable_cache=False)
def predictor(server_url: str, raw_json: str) -> np.ndarray:
    df = json_payload_to_dataframe(raw_json)
    payload = {"dataframe_split": df.to_dict(orient="split")}

    r = requests.post(f"{server_url}/invocations", json=payload, timeout=30)
    r.raise_for_status()

    preds = np.array(r.json())
    preds = np.atleast_1d(preds)   # <-- key fix
    return preds



# ----------------------------
# Pipelines
# ----------------------------
@pipeline(enable_cache=True)
def continuous_deployment_pipeline(max_mse: float = 1.0) -> bool:
    df = ingest_data_step("./data/olist_customers_dataset.csv")
    x_train, x_test, y_train, y_test = clean_data_step(df)

    model = train_model_step(x_train, x_test, y_train, y_test, ModelNameConfig())

    mse, rmse = evaluate_model_step(model, x_test, y_test)

    deploy = deployment_trigger(mse=mse, config=DeploymentTriggerConfig(max_mse=max_mse))
    return deploy


@pipeline(enable_cache=False)
def inference_pipeline(
    model_name: str = "model",
    model_stage_or_version: str = "latest",
    host: str = "127.0.0.1",
    port: int = 5001,
) -> np.ndarray:
    raw = dynamic_importer()

    server_url = start_mlflow_model_server(
        ServeConfig(
            model_name=model_name,
            model_stage_or_version=model_stage_or_version,
            host=host,
            port=port,
            timeout_s=60,
        )
    )

    preds = predictor(server_url=server_url, raw_json=raw)
    return preds
