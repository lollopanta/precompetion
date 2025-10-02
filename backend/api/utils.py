"""Utility functions for the BloomTrack API.

This module provides utility functions for handling geospatial data and raster operations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point, Polygon, box

logger = logging.getLogger(__name__)


def create_geotiff_from_array(
    data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    output_path: str,
    crs: str = "EPSG:4326",
) -> str:
    """Create a GeoTIFF file from a numpy array.

    Args:
        data: 2D numpy array of data.
        bounds: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        output_path: Path to save the GeoTIFF.
        crs: Coordinate reference system.

    Returns:
        Path to the created GeoTIFF.
    """
    # Create transform for GeoTIFF
    transform = from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], data.shape[1], data.shape[0]
    )

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return output_path


def create_tiles_from_geotiff(
    geotiff_path: str, output_dir: str, min_zoom: int = 0, max_zoom: int = 10
) -> str:
    """Create map tiles from a GeoTIFF.

    Args:
        geotiff_path: Path to the GeoTIFF.
        output_dir: Directory to save the tiles.
        min_zoom: Minimum zoom level.
        max_zoom: Maximum zoom level.

    Returns:
        Base URL for the tiles.
    """
    # In a real implementation, we would use a tool like gdal2tiles.py
    # or a tile server like TileServer GL to generate tiles
    # For this example, we'll just return a dummy URL
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"/tiles/{timestamp}/{{z}}/{{x}}/{{y}}.png"


def extract_features_for_location(
    lat: float, lon: float, date: Union[str, datetime]
) -> Dict[str, float]:
    """Extract environmental features for a location and date.

    Args:
        lat: Latitude.
        lon: Longitude.
        date: Date for feature extraction.

    Returns:
        Dictionary of environmental features.
    """
    # In a real implementation, we would query a database or raster files
    # to get the actual environmental features for the location and date
    # For this example, we'll return dummy features

    # Convert date to datetime if it's a string
    if isinstance(date, str):
        date = datetime.fromisoformat(date)

    # Generate some realistic-looking features based on location and date
    # This is just for demonstration purposes
    month = date.month
    # Seasonal factor (northern hemisphere seasons)
    season_factor = np.sin((month - 1) * np.pi / 6)

    # Latitude factor (higher NDVI near equator)
    lat_factor = 1.0 - abs(lat) / 90.0

    # Random variation
    random_factor = np.random.normal(0, 0.1)

    # Calculate NDVI (higher in growing season, lower in winter)
    ndvi = 0.3 + 0.4 * season_factor * lat_factor + random_factor
    ndvi = max(0.0, min(1.0, ndvi))  # Clamp to [0, 1]

    # Calculate EVI (correlated with NDVI but with some differences)
    evi = ndvi * 0.9 + np.random.normal(0, 0.05)
    evi = max(0.0, min(1.0, evi))  # Clamp to [0, 1]

    # Temperature varies with latitude and season
    base_temp = 15.0  # Base temperature in Celsius
    temp_lat_factor = -0.4 * abs(lat)  # Cooler at high latitudes
    temp_season_factor = 15.0 * season_factor  # Seasonal variation
    temperature = base_temp + temp_lat_factor + temp_season_factor + np.random.normal(0, 3.0)

    # Precipitation (higher in tropics and during rainy seasons)
    precip_lat_factor = 100.0 * (1.0 - abs(lat - 15) / 75.0)  # Higher near tropics
    precip_season_factor = 50.0 * season_factor  # Seasonal variation
    precipitation = max(
        0.0, precip_lat_factor + precip_season_factor + np.random.normal(0, 20.0)
    )

    # Elevation (random but consistent for the same location)
    # Use the location coordinates to seed a random generator
    elevation_seed = int(abs(lat * 1000) + abs(lon * 1000))
    np.random.seed(elevation_seed)
    elevation = np.random.uniform(0, 2000)

    # Soil moisture (correlated with precipitation but with lag)
    soil_moisture = 0.2 + 0.6 * (precipitation / 200.0) + np.random.normal(0, 0.05)
    soil_moisture = max(0.0, min(1.0, soil_moisture))  # Clamp to [0, 1]

    # Reset random seed
    np.random.seed(None)

    return {
        "ndvi": float(ndvi),
        "evi": float(evi),
        "temperature": float(temperature),
        "precipitation": float(precipitation),
        "elevation": float(elevation),
        "soil_moisture": float(soil_moisture),
    }


def extract_features_for_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    date: Union[str, datetime],
    resolution: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract environmental features for a bounding box.

    Args:
        min_lat: Minimum latitude.
        min_lon: Minimum longitude.
        max_lat: Maximum latitude.
        max_lon: Maximum longitude.
        date: Date for feature extraction.
        resolution: Grid resolution in degrees.

    Returns:
        Tuple of (features, lats, lons).
    """
    # Create a grid of points within the bounding box
    lons = np.arange(min_lon, max_lon, resolution)
    lats = np.arange(min_lat, max_lat, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten grids for feature extraction
    points = np.vstack([lon_grid.flatten(), lat_grid.flatten()]).T

    # Extract features for each point
    features = []
    for point in points:
        lon, lat = point
        point_features = extract_features_for_location(lat, lon, date)
        features.append(list(point_features.values()))

    # Convert to numpy array
    features_array = np.array(features)

    return features_array, lats, lons