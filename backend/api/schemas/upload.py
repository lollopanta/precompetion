"""Schema definitions for file upload functionality."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ValidationError(BaseModel):
    """Model for validation errors."""
    field: str
    message: str


class UploadResponse(BaseModel):
    """Response model for file upload."""
    filename: str
    size: int
    content_type: str
    success: bool
    message: str
    errors: Optional[List[ValidationError]] = None


class UploadListResponse(BaseModel):
    """Response model for listing uploaded files."""
    files: List[UploadResponse]
    count: int


class FileMetadata(BaseModel):
    """Model for file metadata."""
    filename: str
    original_filename: str
    size: int
    content_type: str
    upload_date: str
    path: str
    description: Optional[str] = None
    validation_status: str = "valid"
    validation_errors: Optional[List[ValidationError]] = None


class FileValidationSettings(BaseModel):
    """Settings for file validation."""
    max_size_mb: float = Field(10.0, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        ["csv", "json", "geojson", "tif", "tiff", "jpg", "jpeg", "png", "txt", "xlsx", "xls"],
        description="List of allowed file extensions"
    )
    allowed_mime_types: List[str] = Field(
        [
            "text/csv", 
            "application/json", 
            "application/geo+json",
            "image/tiff", 
            "image/jpeg", 
            "image/png", 
            "text/plain",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ],
        description="List of allowed MIME types"
    )