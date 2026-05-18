import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "iris-random-forest"

client = MlflowClient()


def get_model_version_by_alias(alias):
    try:
        model_version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=alias
        )
        return model_version
    except Exception:
        return None


def get_accuracy(model_version):
    run = client.get_run(model_version.run_id)
    return run.data.metrics.get("accuracy", 0)


if __name__ == "__main__":
    candidate = get_model_version_by_alias("candidate")
    production = get_model_version_by_alias("production")

    if candidate is None:
        raise Exception("No candidate model found")

    candidate_accuracy = get_accuracy(candidate)

    print(f"Candidate version: {candidate.version}")
    print(f"Candidate accuracy: {candidate_accuracy}")

    if production is None:
        print("No production model exists yet. Promoting candidate to production.")

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=candidate.version
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=candidate.version,
            key="stage",
            value="production"
        )

        print(f"Version {candidate.version} promoted to production")
        exit(0)

    production_accuracy = get_accuracy(production)

    print(f"Production version: {production.version}")
    print(f"Production accuracy: {production_accuracy}")

    if candidate_accuracy > production_accuracy:
        print("Candidate is better. Promoting to production.")

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=candidate.version
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=candidate.version,
            key="stage",
            value="production"
        )

        print(f"Version {candidate.version} promoted to production")
    else:
        print("Candidate is not better. Keeping current production model.")