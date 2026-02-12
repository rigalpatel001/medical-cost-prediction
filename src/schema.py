from pydantic import BaseModel, Field, field_validator
from typing import Literal


class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: Literal["male", "female"]
    bmi: float = Field(..., ge=10, le=60)
    children: int = Field(..., ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]

    @field_validator("bmi")
    def bmi_reasonable(cls, v):
        if v <= 0:
            raise ValueError("BMI must be positive")
        return v
