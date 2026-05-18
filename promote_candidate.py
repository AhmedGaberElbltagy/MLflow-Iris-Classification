import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "iris-random-forest"

client = MlflowClient()


def get_latest_model_version():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        raise Exception(f"No versions found for model: {MODEL_NAME}")

    latest_version = max([int(v.version) for v in versions])
    return latest_version


if __name__ == "__main__":
    latest_version = get_latest_model_version()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="candidate",
        version=latest_version
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=str(latest_version),
        key="stage",
        value="candidate"
    )

    print(f"Model version {latest_version} marked as candidate")
