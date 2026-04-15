from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any


class JobData(BaseModel):
    job_id: str = Field(..., description="The unique ID of the created job")


class JobCreateResponse(BaseModel):
    message: str = Field(..., description="Status message")
    data: JobData


class FightResult(BaseModel):
    fight_detected: bool = Field(..., description="Fight detected")
    max_score: float = Field(..., description="Maximum score")
    min_score: float = Field(..., description="Minimum score")
    avg_score: float = Field(..., description="Average score")
    fight_frequency: float = Field(..., description="Fight frequency")


class VehiclePlateDetail(BaseModel):
    vehicle_id: int = Field(..., description="Vehicle ID")
    plates: List[str] = Field(..., description="List of plates")
    scores: List[float] = Field(..., description="List of scores")


class JobResult(BaseModel):
    filename: str = Field(..., description="Filename")
    url: str = Field(..., description="URL to the result file")
    detections: Optional[Union[FightResult, List[VehiclePlateDetail]]] = Field(None, description="Detections")


class JobStatusData(BaseModel):
    job_id: str = Field(..., description="The unique ID of the created job")
    status: str = Field(..., description="Status of the job")
    result: Optional[JobResult] = Field(None, description="Result of the job")
    error: Optional[str] = Field(None, description="Error message if any")


class JobStatusResponse(BaseModel):
    data: JobStatusData
