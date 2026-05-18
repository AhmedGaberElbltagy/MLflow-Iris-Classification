import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "iris-random-forest"
MODEL_ALIAS = "production"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = mlflow.pyfunc.load_model(model_uri)

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print(f"Prediction: {prediction}")