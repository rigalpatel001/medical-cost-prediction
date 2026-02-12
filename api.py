from fastapi import FastAPI
import pandas as pd
import logging

from src.save_load import load_model
from src.schema import InsuranceInput

app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = load_model("model.joblib")


@app.get("/")
def root():
    return {"message": "Medical Cost Prediction API is running"}


@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]

        logger.info(f"Prediction made for input: {data.dict()}")

        return {"predicted_cost": round(float(prediction), 2)}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": "Prediction failed"}
