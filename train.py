import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-classification")


def train_model(n_estimators, max_depth):
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris-random-forest"
        )

        print("================================")
        print(f"Run ID: {run.info.run_id}")
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {max_depth}")
        print(f"accuracy: {accuracy}")
        print("Model logged and registered")


if __name__ == "__main__":
    train_model(n_estimators=50, max_depth=3)
    train_model(n_estimators=100, max_depth=5)
    train_model(n_estimators=200, max_depth=10)
