from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# 1. Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="REST API to predict customer churn using a trained Scikit-Learn pipeline.",
    version="1.0.0"
)

# 2. Define the expected incoming data schema
# (These match the columns your pipeline expects)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Global variable to hold the model
model_pipeline = None

# 3. Load the model on startup
@app.on_event("startup")
def load_model():
    global model_pipeline
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "results.joblib")
    
    if os.path.exists(model_path):
        # Depending on how you saved it, you might need to extract the best_estimator
        # Since your evaluate.py loaded 'best_model' from the dict, we will do the same:
        results = joblib.load(model_path)
        model_pipeline = results["best_model"]
        print("✅ Pipeline loaded successfully.")
    else:
        print(f"❌ Model file not found at {model_path}")

# 4. Define the prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Convert the Pydantic model to a Pandas DataFrame (1 row)
        input_data = pd.DataFrame([customer.dict()])
        
        # Make predictions
        prediction = model_pipeline.predict(input_data)[0]
        probability = model_pipeline.predict_proba(input_data)[0][1]
        
        # Format the response
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 5. Root health-check endpoint
@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model_pipeline is not None}