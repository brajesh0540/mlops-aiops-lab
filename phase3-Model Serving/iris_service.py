import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
import pandas as pd

# ✅ Load model and create runner
model_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

# ✅ Define BentoML service with runner
svc = bentoml.Service("iris_service", runners=[model_runner])

# ✅ Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ✅ API endpoint
@svc.api(input=JSON(pydantic_model=IrisInput), output=JSON())
async def predict(input_data: IrisInput):
    df = pd.DataFrame([input_data.dict()])
    result = await model_runner.predict.async_run(df)
    return {"prediction": result[0]}
