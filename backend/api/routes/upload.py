"""API routes for file upload functionality."""
import os
import shutil
import uuid
import mimetypes
from datetime import datetime
from typing import List, Set
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..schemas.upload import UploadResponse, UploadListResponse, FileMetadata, ValidationError

router = APIRouter()

# Directory to store uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "uploads")

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for file metadata (in a real app, this would be in a database)
file_metadata_store = {}

# Default validation settings
DEFAULT_VALIDATION_SETTINGS = {
    "max_size_mb": 10.0,  # 10 MB
    "allowed_extensions": [
        "csv", "json", "geojson", "tif", "tiff", "jpg", "jpeg", "png", 
        "txt", "xlsx", "xls", "pdf", "zip", "gz"
    ],
    "allowed_mime_types": [
        "text/csv", 
        "application/json", 
        "application/geo+json",
        "image/tiff", 
        "image/jpeg", 
        "image/png", 
        "text/plain",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/pdf",
        "application/zip",
        "application/gzip"
    ]
}


def validate_file(file: UploadFile, settings: dict = None) -> tuple[bool, List[ValidationError]]:
    """
    Validate the uploaded file against the validation settings.
    
    Args:
        file: The file to validate
        settings: Validation settings (optional)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if settings is None:
        settings = DEFAULT_VALIDATION_SETTINGS
    
    errors = []
    
    # Check file size
    file.file.seek(0, os.SEEK_END)
    size_bytes = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    size_mb = size_bytes / (1024 * 1024)  # Convert to MB
    if size_mb > settings["max_size_mb"]:
        errors.append(ValidationError(
            field="size",
            message=f"File size exceeds maximum allowed size of {settings['max_size_mb']} MB"
        ))
    
    # Check file extension
    _, ext = os.path.splitext(file.filename)
    if ext.startswith('.'):
        ext = ext[1:]  # Remove the dot
    
    if ext.lower() not in settings["allowed_extensions"]:
        errors.append(ValidationError(
            field="extension",
            message=f"File extension '{ext}' not allowed. Allowed extensions: {', '.join(settings['allowed_extensions'])}"
        ))
    
    # Check MIME type
    content_type = file.content_type
    if content_type not in settings["allowed_mime_types"]:
        errors.append(ValidationError(
            field="content_type",
            message=f"File type '{content_type}' not allowed. Allowed types: {', '.join(settings['allowed_mime_types'])}"
        ))
    
    return len(errors) == 0, errors


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: str = Form(None),
    skip_validation: bool = Form(False)
):
    """
    Upload a file to the server.
    
    Args:
        file: The file to upload
        description: Optional description for the file
        skip_validation: Skip file validation if True
        
    Returns:
        UploadResponse: Information about the uploaded file
    """
    # Validate the file if validation is not skipped
    validation_status = "valid"
    validation_errors = []
    
    if not skip_validation:
        is_valid, errors = validate_file(file)
        if not is_valid:
            validation_status = "invalid"
            validation_errors = errors
            
            # If the file is invalid and we don't want to store invalid files,
            # return an error response immediately
            if len(errors) > 0:
                return UploadResponse(
                    filename=file.filename,
                    size=0,
                    content_type=file.content_type,
                    success=False,
                    message="File validation failed",
                    errors=errors
                )
    
    try:
        # Generate a unique filename to prevent collisions
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Store metadata
        metadata = FileMetadata(
            filename=unique_filename,
            original_filename=file.filename,
            size=os.path.getsize(file_path),
            content_type=file.content_type,
            upload_date=datetime.now().isoformat(),
            path=file_path,
            description=description,
            validation_status=validation_status,
            validation_errors=validation_errors
        )
        file_metadata_store[unique_filename] = metadata
        
        return UploadResponse(
            filename=unique_filename,
            size=metadata.size,
            content_type=metadata.content_type,
            success=True,
            message="File uploaded successfully"
        )
    except Exception as e:
        return UploadResponse(
            filename=file.filename,
            size=0,
            content_type=file.content_type,
            success=False,
            message=f"Error uploading file: {str(e)}"
        )


@router.get("/files", response_model=UploadListResponse)
async def list_files(
    valid_only: bool = Query(False, description="Only return valid files"),
    file_type: str = Query(None, description="Filter by file type/extension")
):
    """
    List all uploaded files with optional filtering.
    
    Args:
        valid_only: If True, only return files that passed validation
        file_type: Filter files by extension (e.g., 'csv', 'json')
        
    Returns:
        UploadListResponse: List of uploaded files
    """
    files = []
    
    for filename, metadata in file_metadata_store.items():
        # Apply filters
        if valid_only and metadata.validation_status != "valid":
            continue
            
        if file_type:
            _, ext = os.path.splitext(metadata.original_filename)
            if ext.startswith('.'):
                ext = ext[1:]  # Remove the dot
            if ext.lower() != file_type.lower():
                continue
        
        files.append(UploadResponse(
            filename=filename,
            size=metadata.size,
            content_type=metadata.content_type,
            success=True,
            message="File retrieved"
        ))
    
    return UploadListResponse(
        files=files,
        count=len(files)
    )


@router.get("/files/{filename}/content")
async def get_file_content(filename: str):
    """
    Get the content of an uploaded file.
    
    Args:
        filename: The unique filename of the file to retrieve
        
    Returns:
        FileResponse: The file content
    """
    if filename not in file_metadata_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get the file path
    file_path = file_metadata_store[filename].path
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine content type
    content_type = file_metadata_store[filename].content_type
    
    # Return the file
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=file_metadata_store[filename].original_filename,
        media_type=content_type
    )


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """
    Delete an uploaded file.
    
    Args:
        filename: The unique filename of the file to delete
        
    Returns:
        JSONResponse: Status of the deletion operation
    """
    if filename not in file_metadata_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Get file path
        file_path = file_metadata_store[filename].path
        
        # Delete the file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove metadata
        del file_metadata_store[filename]
        
        return JSONResponse(
            content={"success": True, "message": "File deleted successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")