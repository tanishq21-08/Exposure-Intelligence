from enum import Enum
from typing import Optional
from pydantic import BaseModel

class FieldType(str, Enum):
    verbatim = "verbatim"
    derived = "derived"

class TextField(BaseModel):
    value: str
    confidence: float
    source: str
    type: FieldType

class NumField(BaseModel):
    value: Optional[float]
    confidence: float
    source: str
    type: FieldType

class Location(BaseModel):
    address: TextField
    tiv_gbp: NumField
    construction: TextField
    occupancy: TextField
    year_built: NumField
    storeys: NumField
    floor_area_sqft: NumField
    sprinklered: TextField

class Portfolio(BaseModel):
    locations: list[Location]