import mlflow
from fastapi import FastAPI
from pydantic import BaseModel


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "iris-random-forest"
MODEL_ALIAS = "production"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Iris MLflow Model API")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


@app.get("/")
def root():
    return {
        "message": "Iris MLflow model API is running",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }


@app.post("/predict")
def predict(input_data: IrisInput):
    sample = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]

    prediction = model.predict(sample)
    predicted_class = int(prediction[0])

    return {
        "prediction_id": predicted_class,
        "prediction_name": class_names[predicted_class]
    }
