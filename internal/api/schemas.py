from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Union


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

class AddCameraRequest(BaseModel):
    name: str = Field(description="Name of the camera")
    url: str = Field(description="URL of the camera stream")
    fight_enabled: bool = Field(description="Enable fight detection")
    vehicle_plate_enabled: bool = Field(description="Enable vehicle plate detection")
    snapshot_enabled: bool = Field(description="Enable snapshot")
    mqtt_enabled: bool = Field(description="Enable MQTT")

class UpdateCameraRequest(BaseModel):
    name: Optional[str] = Field(None, description="Name of the camera")
    url: Optional[str] = Field(None, description="URL of the camera stream")
    fight_enabled: Optional[bool] = Field(None, description="Enable fight detection")
    vehicle_plate_enabled: Optional[bool] = Field(None, description="Enable vehicle plate detection")
    snapshot_enabled: Optional[bool] = Field(None, description="Enable snapshot")
    mqtt_enabled: Optional[bool] = Field(None, description="Enable MQTT")

class CameraResponse(BaseModel):
    id: int
    name: str
    url: str
    fight_enabled: bool
    vehicle_plate_enabled: bool
    snapshot_enabled: bool
    mqtt_enabled: bool

    class Config:
        from_attributes = True

class GenerateApiKeyRequest(BaseModel):
    name: str = Field(description="Name of the API key")
    expires_at: Optional[datetime] = Field(None, description="Expiration date of the API key")

class TokenResponse(BaseModel):
    id: int
    name: str
    token: str
    expires_at: Optional[datetime] = None
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True

class GenericResponse(BaseModel):
    status: str = Field(example="success")
    message: str = Field(example="Face registered successfully")