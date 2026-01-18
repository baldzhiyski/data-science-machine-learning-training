# run_deployment_pipeline.py
from rich import print
import click

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
)
@click.option(
    "--max-mse",
    default=1.0,
    type=float,
    help="Maximum MSE allowed to deploy the model (lower is better).",
)
@click.option(
    "--model-name",
    default="model",
    type=str,
)
@click.option(
    "--model-stage-or-version",
    default="latest",  # or "1" or "Production"
    type=str,
)
def main(config: str, max_mse: float, model_name: str, model_stage_or_version: str):
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    if deploy:
        continuous_deployment_pipeline(max_mse=max_mse)
        print("[green]Deployment pipeline finished (model logged to MLflow).[/green]")

    if predict:
        preds = inference_pipeline(
            model_name=model_name,
            model_stage_or_version=model_stage_or_version,
        )
        print("[cyan]Predictions:[/cyan]", preds)


if __name__ == "__main__":
    main()
