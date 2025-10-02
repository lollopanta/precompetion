"""Main FastAPI application for BloomTrack.

This module sets up the FastAPI application and includes all routes.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import rasterio
import xarray as xr
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rasterio.transform import from_bounds

from backend.api.routes import router as api_router
from backend.models.inference.bloom_predictor import BloomPredictor, load_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BloomTrack API",
    description="API for global plant bloom detection and prediction",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Global predictor instance
predictor: Optional[BloomPredictor] = None


# Models for API requests and responses
class LocationQuery(BaseModel):
    """Model for location-based bloom prediction query."""

    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    date: str = Field(
        ..., description="Date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    include_forecast: bool = Field(
        False, description="Whether to include time series forecast"
    )
    forecast_days: int = Field(
        30, description="Number of days to forecast (if include_forecast is True)"
    )


class BoundingBoxQuery(BaseModel):
    """Model for bounding box bloom prediction query."""

    min_lat: float = Field(..., description="Minimum latitude")
    min_lon: float = Field(..., description="Minimum longitude")
    max_lat: float = Field(..., description="Maximum latitude")
    max_lon: float = Field(..., description="Maximum longitude")
    date: str = Field(
        ..., description="Date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )
    resolution: float = Field(
        0.01, description="Grid resolution in degrees (lower is more detailed)"
    )


class BloomPrediction(BaseModel):
    """Model for bloom prediction response."""

    latitude: float
    longitude: float
    prediction_date: str
    bloom_probability: float
    is_blooming: bool
    bloom_intensity: Optional[float] = None
    forecast: Optional[List[Dict[str, Union[str, float]]]] = None
    metadata: Dict[str, Union[str, float]]


class GridPredictionResponse(BaseModel):
    """Model for grid prediction response."""

    bounds: Dict[str, float]
    resolution: float
    prediction_date: str
    raster_url: str
    tile_url: str
    metadata: Dict[str, Union[str, float]]


# Dependency to get predictor
def get_predictor() -> BloomPredictor:
    """Get the bloom predictor instance.

    Returns:
        BloomPredictor instance.
    """
    global predictor
    if predictor is None:
        try:
            # Load predictor from models directory
            model_dir = Path("./models")
            predictor = load_predictor(str(model_dir))
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to load prediction models"
            )
    return predictor


@app.get("/")
async def root():
    """Root endpoint.

    Returns:
        Welcome message.
    """
    return {"message": "Welcome to BloomTrack API"}


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict/location", response_model=BloomPrediction)
async def predict_location(
    query: LocationQuery, predictor: BloomPredictor = Depends(get_predictor)
):
    """Predict bloom metrics for a specific location.

    Args:
        query: Location query parameters.
        predictor: BloomPredictor instance.

    Returns:
        Bloom prediction for the location.
    """
    try:
        # In a real implementation, we would fetch environmental features
        # from a database or compute them from raster data
        # For this example, we'll use dummy features
        environmental_features = {
            "ndvi": 0.65,
            "evi": 0.58,
            "precipitation": 120.5,
            "temperature": 22.3,
            "elevation": 250.0,
            "soil_moisture": 0.35,
        }

        # Make prediction
        result = predictor.predict_location(
            lat=query.latitude,
            lon=query.longitude,
            date=query.date,
            environmental_features=environmental_features,
        )

        # Create response
        response = BloomPrediction(
            latitude=query.latitude,
            longitude=query.longitude,
            prediction_date=result["prediction_date"],
            bloom_probability=result.get("bloom_probability", 0.0),
            is_blooming=result.get("is_blooming", False),
            bloom_intensity=result.get("spatial_bloom_intensity"),
            forecast=None,  # Would be populated with time series forecast
            metadata={
                "prediction_timestamp": result["prediction_timestamp"],
                "model_version": "0.1.0",
            },
        )

        # Add forecast if requested
        if query.include_forecast and query.forecast_days > 0:
            # In a real implementation, we would generate a forecast
            # For this example, we'll use dummy forecast data
            forecast = [
                {
                    "date": (
                        datetime.fromisoformat(query.date) + 
                        timedelta(days=i)
                    ).isoformat(),
                    "bloom_probability": min(
                        max(result.get("bloom_probability", 0.5) + 
                            (np.sin(i / 10) * 0.2), 0.0), 1.0
                    ),
                }
                for i in range(1, query.forecast_days + 1)
            ]
            response.forecast = forecast

        return response

    except Exception as e:
        logger.error(f"Error predicting for location: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/bbox", response_model=GridPredictionResponse)
async def predict_bbox(
    query: BoundingBoxQuery, predictor: BloomPredictor = Depends(get_predictor)
):
    """Predict bloom metrics for a bounding box.

    Args:
        query: Bounding box query parameters.
        predictor: BloomPredictor instance.

    Returns:
        Grid prediction for the bounding box.
    """
    try:
        # In a real implementation, we would fetch features for the bounding box
        # For this example, we'll generate dummy data
        bounds = (query.min_lon, query.min_lat, query.max_lon, query.max_lat)
        
        # Create a grid of points within the bounding box
        lons = np.arange(bounds[0], bounds[2], query.resolution)
        lats = np.arange(bounds[1], bounds[3], query.resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Flatten grids for prediction
        points = np.vstack([lon_grid.flatten(), lat_grid.flatten()]).T
        
        # Generate dummy features for each point
        n_points = len(points)
        n_features = 6  # Example: NDVI, EVI, precipitation, temperature, elevation, soil moisture
        features = np.random.rand(n_points, n_features)
        
        # In a real implementation, we would use the predictor to generate a grid prediction
        # For this example, we'll create a dummy DataArray
        prediction_grid = np.random.rand(len(lats), len(lons)) * 0.8 + 0.1
        
        # Create DataArray
        da = xr.DataArray(
            data=prediction_grid,
            dims=["latitude", "longitude"],
            coords={
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "long_name": "Bloom Probability",
                "units": "",
                "resolution": query.resolution,
            },
        )
        
        # Save as GeoTIFF
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        raster_filename = f"bloom_prediction_{timestamp}.tif"
        raster_path = output_dir / raster_filename
        
        # Create transform for GeoTIFF
        transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3], 
            prediction_grid.shape[1], prediction_grid.shape[0]
        )
        
        # Write GeoTIFF
        with rasterio.open(
            str(raster_path),
            "w",
            driver="GTiff",
            height=prediction_grid.shape[0],
            width=prediction_grid.shape[1],
            count=1,
            dtype=prediction_grid.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(prediction_grid, 1)
        
        # In a real implementation, we would also generate map tiles
        # For this example, we'll just provide a dummy URL
        tile_url = f"/tiles/{timestamp}/{{z}}/{{x}}/{{y}}.png"
        
        # Create response
        response = GridPredictionResponse(
            bounds={
                "min_lat": query.min_lat,
                "min_lon": query.min_lon,
                "max_lat": query.max_lat,
                "max_lon": query.max_lon,
            },
            resolution=query.resolution,
            prediction_date=query.date,
            raster_url=f"/rasters/{raster_filename}",
            tile_url=tile_url,
            metadata={
                "prediction_timestamp": datetime.now().isoformat(),
                "model_version": "0.1.0",
                "cell_count": len(lons) * len(lats),
            },
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error predicting for bounding box: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def get_model_info(predictor: BloomPredictor = Depends(get_predictor)):
    """Get information about the loaded models.

    Args:
        predictor: BloomPredictor instance.

    Returns:
        Model information.
    """
    models_info = {
        "classifier_loaded": predictor.classifier is not None,
        "forecaster_loaded": predictor.forecaster is not None,
        "spatial_model_loaded": predictor.spatial_model is not None,
        "feature_scaler_loaded": predictor.feature_scaler is not None,
    }
    
    return models_info


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)