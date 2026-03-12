import joblib
from fastapi import (
    FastAPI, APIRouter, Depends, HTTPException,
    Request, Response, BackgroundTasks
)
from pydantic import BaseModel
from pydantic_settings import BaseSettings

import pandas as pd
import time

bundle = joblib.load("model/model_bundle.pkl")
#Configuration
class Settings(BaseSettings):
    model_name: str = "ibm_hr_attrition_model"
    model_version: str = "v2.0"
#    model_path: str = bundle["model"]
#    scaler_path: str = bundle["scaler"]
#   columns_path: str = bundle["scaler"]

settings = Settings()

#Input_schema

class EmployeeInput(BaseModel):
    Age: int
    MonthlyIncome: float
    JobSatisfaction: int
    YearsAtCompany: int
    OverTime: int

#Output_schema

class PredictionOutput(BaseModel):
    attrition: str
    probability: float
    model_version: str

class AttritionModelService:
    def __init__(self):
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.columns = bundle["features"]

    def preprocess(self, data: EmployeeInput):
        df = pd.DataFrame([data.dict()])

        # Ensure correct column order
        df = df.reindex(columns=self.columns, fill_value=0)

        # Scale numeric features
        df_scaled = self.scaler.transform(df)

        return df_scaled

    def predict(self, data: EmployeeInput):
        X = self.preprocess(data)

        prob = self.model.predict_proba(X)[0][1]
        label = "Yes" if prob >= 0.5 else "No"

        return label, float(prob)

# Dependency injection
model_service = AttritionModelService()

def get_model_service():
    return model_service

#Background logging

def log_prediction(input_data: EmployeeInput, result: str, prob: float):
    time.sleep(0.1)
    print(
        f"[LOG] input={input_data.dict()} "
        f"→ attrition={result}, prob={prob:.3f}"
    )

#Fast API app
app = FastAPI(title="IBM HR Attrition ML API", version="2.0")
router = APIRouter(
    prefix="/api",
    tags=["Attrition Model"]
)

#Prediction

@router.post("/predict", response_model=PredictionOutput)
def predict_attrition(
    employee: EmployeeInput,
    request: Request,
    response: Response,
    bg: BackgroundTasks,
    model: AttritionModelService = Depends(get_model_service)
):
    if employee.Age < 18:
        raise HTTPException(status_code=400, detail="Invalid age")

    label, prob = model.predict(employee)

    response.headers["X-Model-Name"] = settings.model_name

    bg.add_task(log_prediction, employee, label, prob)

    return PredictionOutput(
        attrition=label,
        probability=round(prob, 3),
        model_version=settings.model_version
    )
@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model": settings.model_name,
        "version": settings.model_version
    }

#
@app.get("/")
def home():
    return {"message": "IBM HR Attrition API is running"}

router = APIRouter(prefix="/api")