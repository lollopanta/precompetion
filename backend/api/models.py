"""API models and schemas for BloomTrack API.

This module provides Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class LocationQuery(BaseModel):
    """Model for location-based bloom prediction query."""

    latitude: float = Field(
        ..., description="Latitude in decimal degrees", ge=-90, le=90
    )
    longitude: float = Field(
        ..., description="Longitude in decimal degrees", ge=-180, le=180
    )
    date: str = Field(
        ..., description="Date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    include_forecast: bool = Field(
        False, description="Whether to include time series forecast"
    )
    forecast_days: int = Field(
        30, description="Number of days to forecast (if include_forecast is True)",
        ge=1, le=365
    )

    @validator("date")
    def validate_date(cls, v):
        """Validate that the date is in ISO format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")


class BoundingBoxQuery(BaseModel):
    """Model for bounding box bloom prediction query."""

    min_lat: float = Field(..., description="Minimum latitude", ge=-90, le=90)
    min_lon: float = Field(..., description="Minimum longitude", ge=-180, le=180)
    max_lat: float = Field(..., description="Maximum latitude", ge=-90, le=90)
    max_lon: float = Field(..., description="Maximum longitude", ge=-180, le=180)
    date: str = Field(
        ..., description="Date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    resolution: float = Field(
        0.01, description="Grid resolution in degrees (lower is more detailed)",
        gt=0, le=1
    )

    @validator("date")
    def validate_date(cls, v):
        """Validate that the date is in ISO format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

    @validator("max_lat")
    def validate_lat_bounds(cls, v, values):
        """Validate that max_lat is greater than min_lat."""
        if "min_lat" in values and v <= values["min_lat"]:
            raise ValueError("max_lat must be greater than min_lat")
        return v

    @validator("max_lon")
    def validate_lon_bounds(cls, v, values):
        """Validate that max_lon is greater than min_lon."""
        if "min_lon" in values and v <= values["min_lon"]:
            raise ValueError("max_lon must be greater than min_lon")
        return v


class ForecastPoint(BaseModel):
    """Model for a single forecast point."""

    date: str
    bloom_probability: float
    bloom_intensity: Optional[float] = None


class BloomPrediction(BaseModel):
    """Model for bloom prediction response."""

    latitude: float
    longitude: float
    prediction_date: str
    bloom_probability: float
    is_blooming: bool
    bloom_intensity: Optional[float] = None
    forecast: Optional[List[ForecastPoint]] = None
    metadata: Dict[str, Union[str, float]]


class GridPredictionResponse(BaseModel):
    """Model for grid prediction response."""

    bounds: Dict[str, float]
    resolution: float
    prediction_date: str
    raster_url: str
    tile_url: str
    metadata: Dict[str, Union[str, float]]


class ModelInfo(BaseModel):
    """Model for model information response."""

    classifier_loaded: bool
    forecaster_loaded: bool
    spatial_model_loaded: bool
    feature_scaler_loaded: bool
    model_version: str = "0.1.0"
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())