from fastapi import APIRouter, Depends, HTTPException
from app.schemas.employee_schema import EmployeeInput, PredictionOutput
from app.services.model_service import AttritionModelService
from app.config.settings import settings

router = APIRouter(prefix="/api", tags=["Attrition Model"])

model_service = AttritionModelService()

@router.post("/predict", response_model=PredictionOutput)
def predict_attrition(employee: EmployeeInput):

    if employee.Age < 18:
        raise HTTPException(status_code=400, detail="Invalid age")

    label, prob = model_service.predict(employee)

    return PredictionOutput(
        attrition=label,
        probability=round(prob,3),
        model_version=settings.model_version
    )


@router.get("/health")
def health():



    return {
        "status": "healthy",
        "model": settings.model_name,
        "version": settings.model_version
    }