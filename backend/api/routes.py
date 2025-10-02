"""API routes for BloomTrack.

This module defines the API routes for the BloomTrack application.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

# Import upload router
from backend.api.routes.upload import router as upload_router

from backend.api.models import (
    BloomPrediction,
    BoundingBoxQuery,
    ForecastPoint,
    GridPredictionResponse,
    LocationQuery,
    ModelInfo,
)
from backend.api.utils import (
    create_geotiff_from_array,
    create_tiles_from_geotiff,
    extract_features_for_bbox,
    extract_features_for_location,
)
from backend.models.inference.bloom_predictor import BloomPredictor, load_predictor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Include upload router
router.include_router(upload_router, prefix="/upload", tags=["upload"])

# Initialize predictor
predictor: Optional[BloomPredictor] = None


def get_predictor() -> BloomPredictor:
    """Get or initialize the bloom predictor.

    Returns:
        Initialized BloomPredictor instance.
    """
    global predictor
    if predictor is None:
        try:
            predictor = load_predictor()
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            # For demo purposes, we'll continue without a predictor
            # In production, you might want to raise an exception here
            predictor = None

    return predictor


@router.get("/", summary="Root endpoint")
async def root() -> Dict[str, str]:
    """Root endpoint.

    Returns:
        Welcome message.
    """
    return {"message": "Welcome to BloomTrack API"}


@router.get("/health", summary="Health check endpoint")
async def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.post(
    "/predict/location", response_model=BloomPrediction, summary="Predict bloom for a location"
)
async def predict_location(
    query: LocationQuery, predictor: BloomPredictor = Depends(get_predictor)
) -> BloomPrediction:
    """Predict bloom for a specific location and date.

    Args:
        query: Location query parameters.
        predictor: Bloom predictor instance.

    Returns:
        Bloom prediction for the location.
    """
    # Extract features for the location
    features = extract_features_for_location(query.latitude, query.longitude, query.date)

    # Make prediction
    try:
        if predictor is not None:
            # Use the actual predictor if available
            bloom_prob = predictor.predict_bloom_probability(
                latitude=query.latitude,
                longitude=query.longitude,
                date=query.date,
                features=features,
            )
            is_blooming = predictor.is_blooming(
                latitude=query.latitude,
                longitude=query.longitude,
                date=query.date,
                features=features,
            )
            bloom_intensity = predictor.predict_bloom_intensity(
                latitude=query.latitude,
                longitude=query.longitude,
                date=query.date,
                features=features,
            )
        else:
            # Generate dummy prediction if predictor is not available
            # This is just for demonstration purposes
            bloom_prob = features["ndvi"] * 0.8 + features["evi"] * 0.2
            is_blooming = bloom_prob > 0.6
            bloom_intensity = bloom_prob * 10 if is_blooming else 0.0

        # Create response
        response = BloomPrediction(
            latitude=query.latitude,
            longitude=query.longitude,
            prediction_date=query.date,
            bloom_probability=float(bloom_prob),
            is_blooming=bool(is_blooming),
            bloom_intensity=float(bloom_intensity),
            metadata=features,
        )

        # Add forecast if requested
        if query.include_forecast:
            forecast_points = []
            prediction_date = datetime.fromisoformat(query.date)

            for i in range(1, query.forecast_days + 1):
                forecast_date = prediction_date + timedelta(days=i)
                forecast_features = extract_features_for_location(
                    query.latitude, query.longitude, forecast_date
                )

                if predictor is not None:
                    # Use the actual predictor if available
                    forecast_prob = predictor.forecast_bloom(
                        latitude=query.latitude,
                        longitude=query.longitude,
                        start_date=query.date,
                        target_date=forecast_date.isoformat(),
                        features=forecast_features,
                    )
                    forecast_intensity = predictor.predict_bloom_intensity(
                        latitude=query.latitude,
                        longitude=query.longitude,
                        date=forecast_date.isoformat(),
                        features=forecast_features,
                    )
                else:
                    # Generate dummy forecast if predictor is not available
                    base_prob = features["ndvi"] * 0.8 + features["evi"] * 0.2
                    # Add some seasonal variation
                    season_factor = np.sin((forecast_date.month - 1) * np.pi / 6)
                    forecast_prob = base_prob + 0.1 * season_factor * (i / query.forecast_days)
                    forecast_prob = max(0.0, min(1.0, forecast_prob))  # Clamp to [0, 1]
                    forecast_intensity = forecast_prob * 10 if forecast_prob > 0.6 else 0.0

                forecast_points.append(
                    ForecastPoint(
                        date=forecast_date.isoformat(),
                        bloom_probability=float(forecast_prob),
                        bloom_intensity=float(forecast_intensity),
                    )
                )

            response.forecast = forecast_points

        return response

    except Exception as e:
        logger.error(f"Error predicting bloom for location: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post(
    "/predict/bbox", response_model=GridPredictionResponse, summary="Predict bloom for a bounding box"
)
async def predict_bbox(
    query: BoundingBoxQuery, predictor: BloomPredictor = Depends(get_predictor)
) -> GridPredictionResponse:
    """Predict bloom for a bounding box.

    Args:
        query: Bounding box query parameters.
        predictor: Bloom predictor instance.

    Returns:
        Grid prediction response with URLs to GeoTIFF and tiles.
    """
    try:
        # Extract features for the bounding box
        features_array, lats, lons = extract_features_for_bbox(
            query.min_lat,
            query.min_lon,
            query.max_lat,
            query.max_lon,
            query.date,
            query.resolution,
        )

        # Reshape features for prediction
        num_features = features_array.shape[1]
        height = len(lats)
        width = len(lons)
        features_grid = features_array.reshape(height, width, num_features)

        # Make prediction
        if predictor is not None and hasattr(predictor, "predict_spatial"):
            # Use the actual predictor if available
            bloom_prob_grid = predictor.predict_spatial(
                min_lat=query.min_lat,
                min_lon=query.min_lon,
                max_lat=query.max_lat,
                max_lon=query.max_lon,
                date=query.date,
                resolution=query.resolution,
            )
        else:
            # Generate dummy prediction if predictor is not available
            # This is just for demonstration purposes
            ndvi_index = 0  # Assuming NDVI is the first feature
            evi_index = 1   # Assuming EVI is the second feature
            ndvi_grid = features_grid[:, :, ndvi_index]
            evi_grid = features_grid[:, :, evi_index]
            bloom_prob_grid = ndvi_grid * 0.8 + evi_grid * 0.2

        # Create output directory for GeoTIFF and tiles
        output_dir = Path("data/output/predictions")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Create GeoTIFF
        geotiff_path = str(output_dir / f"bloom_prediction_{timestamp}.tif")
        create_geotiff_from_array(
            bloom_prob_grid,
            (query.min_lon, query.min_lat, query.max_lon, query.max_lat),
            geotiff_path,
        )

        # Create tiles
        tiles_dir = str(output_dir / f"tiles_{timestamp}")
        tile_url = create_tiles_from_geotiff(geotiff_path, tiles_dir)

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
            raster_url=f"/api/rasters/{os.path.basename(geotiff_path)}",
            tile_url=tile_url,
            metadata={
                "mean_probability": float(np.mean(bloom_prob_grid)),
                "max_probability": float(np.max(bloom_prob_grid)),
                "min_probability": float(np.min(bloom_prob_grid)),
                "std_probability": float(np.std(bloom_prob_grid)),
                "prediction_algorithm": "spatial_bloom_model" if predictor is not None else "dummy_model",
            },
        )

        return response

    except Exception as e:
        logger.error(f"Error predicting bloom for bounding box: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get(
    "/rasters/{filename}", summary="Get a GeoTIFF raster file"
)
async def get_raster(filename: str) -> FileResponse:
    """Get a GeoTIFF raster file.

    Args:
        filename: Name of the GeoTIFF file.

    Returns:
        GeoTIFF file.
    """
    file_path = Path(f"data/output/predictions/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Raster file not found")

    return FileResponse(
        str(file_path),
        media_type="image/tiff",
        filename=filename,
    )


@router.get("/model/info", response_model=ModelInfo, summary="Get model information")
async def get_model_info(predictor: BloomPredictor = Depends(get_predictor)) -> ModelInfo:
    """Get information about the loaded models.

    Args:
        predictor: Bloom predictor instance.

    Returns:
        Model information.
    """
    if predictor is None:
        return ModelInfo(
            classifier_loaded=False,
            forecaster_loaded=False,
            spatial_model_loaded=False,
            feature_scaler_loaded=False,
            model_version="0.1.0",
            last_updated=datetime.now().isoformat(),
        )

    return ModelInfo(
        classifier_loaded=predictor.classifier is not None,
        forecaster_loaded=predictor.forecaster is not None,
        spatial_model_loaded=predictor.spatial_model is not None,
        feature_scaler_loaded=predictor.feature_scaler is not None,
        model_version=predictor.version,
        last_updated=predictor.last_updated,
    )