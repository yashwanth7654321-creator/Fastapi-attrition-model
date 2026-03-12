from pydantic import BaseModel

class EmployeeInput(BaseModel):
    Age: int
    MonthlyIncome: float
    JobSatisfaction: int
    YearsAtCompany: int
    OverTime: int


class PredictionOutput(BaseModel):
    attrition: str
    probability: float
    model_version: str