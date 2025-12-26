from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MatchBBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class MatchResponse(BaseModel):
    matched: bool
    confidence: float  # 0.0 - 1.0
    timestamp_ms: int  # the requested time in the recording that was checked
    bbox: Optional[MatchBBox] = None  # bounding box in the extracted frame (if available)
    message: Optional[str] = None
    processed_at: datetime