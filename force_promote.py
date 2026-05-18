import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "iris-random-forest"
VERSION = "6"

client = MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=VERSION
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=VERSION,
    key="stage",
    value="production"
)

print(f"Version {VERSION} is now production")
