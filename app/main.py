from fastapi import FastAPI
from app.routers.prediction_router import router

app = FastAPI(title="IBM HR Attrition ML API", version="2.0")

app.include_router(router)

print("API is running fast")
