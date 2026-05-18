import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "iris-random-forest"

client = MlflowClient()


def get_best_model_version():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    best_version = None
    best_accuracy = -1

    for version in versions:
        run_id = version.run_id
        run = client.get_run(run_id)

        accuracy = run.data.metrics.get("accuracy", 0)

        print(f"Version: {version.version}, Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_version = version.version

    return best_version, best_accuracy


if __name__ == "__main__":
    best_version, best_accuracy = get_best_model_version()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=best_version
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=best_version,
        key="stage",
        value="production"
    )

    print(f"Promoted version {best_version} to production")
    print(f"Accuracy: {best_accuracy}")