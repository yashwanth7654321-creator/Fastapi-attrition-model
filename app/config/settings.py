from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "ibm_hr_attrition_model"
    model_version: str = "v2.0"

settings = Settings()