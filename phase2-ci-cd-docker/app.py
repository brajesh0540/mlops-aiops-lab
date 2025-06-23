from fastapi import FastAPI
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
model = joblib.load("model.pkl")

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": pred}
