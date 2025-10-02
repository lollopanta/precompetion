"""Environmental feature extraction module.

This module provides utilities for extracting environmental features from geospatial data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import box, mapping

logger = logging.getLogger(__name__)


class EnvironmentalFeatureExtractor:
    """Class for extracting environmental features from geospatial data."""

    def __init__(self):
        """Initialize the environmental feature extractor."""
        self.dem_source = None
        self.soil_source = None
        self.climate_source = None

    def load_dem(self, dem_path: str) -> None:
        """Load Digital Elevation Model (DEM) data.

        Args:
            dem_path: Path to the DEM file (GeoTIFF format).
        """
        logger.info(f"Loading DEM from {dem_path}")
        self.dem_source = rasterio.open(dem_path)

    def load_soil_data(self, soil_path: str) -> None:
        """Load soil data.

        Args:
            soil_path: Path to the soil data file (GeoTIFF or vector format).
        """
        logger.info(f"Loading soil data from {soil_path}")
        # This is a simplified implementation
        # In a real application, this would handle different soil data formats
        if soil_path.endswith(".tif") or soil_path.endswith(".tiff"):
            self.soil_source = rasterio.open(soil_path)
        else:
            self.soil_source = gpd.read_file(soil_path)

    def load_climate_data(self, climate_path: str) -> None:
        """Load climate data.

        Args:
            climate_path: Path to the climate data file (NetCDF or similar).
        """
        logger.info(f"Loading climate data from {climate_path}")
        # This is a simplified implementation
        # In a real application, this would handle different climate data formats
        self.climate_source = xr.open_dataset(climate_path)

    def extract_elevation_features(
        self, bbox: Tuple[float, float, float, float], resolution: Optional[Tuple[float, float]] = None
    ) -> Dict[str, xr.DataArray]:
        """Extract elevation-related features for a bounding box.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat).
            resolution: Output resolution (x_res, y_res) in target CRS units.
                        If None, will use the DEM's native resolution.

        Returns:
            Dictionary of elevation features as xarray DataArrays.
        """
        if self.dem_source is None:
            logger.error("DEM not loaded. Call load_dem() first.")
            return {}

        logger.info(f"Extracting elevation features for bbox {bbox}")

        # Create a geometry from the bounding box
        geom = box(*bbox)

        # Read the DEM data for the bounding box
        window = rasterio.windows.from_bounds(*bbox, transform=self.dem_source.transform)
        dem_data = self.dem_source.read(1, window=window)

        # Get the transform for the window
        window_transform = rasterio.windows.transform(window, self.dem_source.transform)

        # Create coordinates for the output arrays
        height, width = dem_data.shape
        x_coords = np.linspace(bbox[0], bbox[2], width)
        y_coords = np.linspace(bbox[3], bbox[1], height)  # Note: y decreases as index increases

        # Create an xarray DataArray for elevation
        elevation = xr.DataArray(
            dem_data,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="elevation",
            attrs={
                "long_name": "Elevation",
                "units": "meters",
                "crs": str(self.dem_source.crs),
            },
        )

        # Calculate slope and aspect (simplified implementation)
        # In a real application, this would use more sophisticated algorithms
        # such as those provided by richdem or similar libraries
        dy, dx = np.gradient(dem_data)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # Create xarray DataArrays for slope and aspect
        slope_da = xr.DataArray(
            slope,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="slope",
            attrs={
                "long_name": "Slope",
                "units": "degrees",
                "crs": str(self.dem_source.crs),
            },
        )

        aspect_da = xr.DataArray(
            aspect,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="aspect",
            attrs={
                "long_name": "Aspect",
                "units": "degrees",
                "crs": str(self.dem_source.crs),
            },
        )

        return {
            "elevation": elevation,
            "slope": slope_da,
            "aspect": aspect_da,
        }

    def extract_soil_features(
        self, bbox: Tuple[float, float, float, float], resolution: Optional[Tuple[float, float]] = None
    ) -> Dict[str, xr.DataArray]:
        """Extract soil-related features for a bounding box.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat).
            resolution: Output resolution (x_res, y_res) in target CRS units.
                        If None, will use the soil data's native resolution.

        Returns:
            Dictionary of soil features as xarray DataArrays.
        """
        if self.soil_source is None:
            logger.error("Soil data not loaded. Call load_soil_data() first.")
            return {}

        logger.info(f"Extracting soil features for bbox {bbox}")

        # This is a simplified implementation
        # In a real application, this would handle different soil data formats and properties
        # For demonstration, we'll return placeholder data

        # Create a geometry from the bounding box
        geom = box(*bbox)

        # Create placeholder soil data
        # In a real application, this would extract actual soil properties
        soil_moisture = np.random.uniform(0.1, 0.5, (100, 100))
        soil_ph = np.random.uniform(5.5, 7.5, (100, 100))

        # Create coordinates for the output arrays
        x_coords = np.linspace(bbox[0], bbox[2], 100)
        y_coords = np.linspace(bbox[3], bbox[1], 100)

        # Create xarray DataArrays for soil properties
        soil_moisture_da = xr.DataArray(
            soil_moisture,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="soil_moisture",
            attrs={
                "long_name": "Soil Moisture",
                "units": "volumetric fraction",
            },
        )

        soil_ph_da = xr.DataArray(
            soil_ph,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="soil_ph",
            attrs={
                "long_name": "Soil pH",
                "units": "pH",
            },
        )

        return {
            "soil_moisture": soil_moisture_da,
            "soil_ph": soil_ph_da,
        }

    def extract_climate_features(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, xr.DataArray]:
        """Extract climate-related features for a bounding box and time range.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat).
            time_range: Time range as (start_date, end_date) in ISO format.
                        If None, will use all available times.

        Returns:
            Dictionary of climate features as xarray DataArrays.
        """
        if self.climate_source is None:
            logger.error("Climate data not loaded. Call load_climate_data() first.")
            return {}

        logger.info(f"Extracting climate features for bbox {bbox} and time range {time_range}")

        # This is a simplified implementation
        # In a real application, this would extract actual climate data for the specified region and time
        # For demonstration, we'll return placeholder data

        # Create placeholder climate data
        # In a real application, this would extract actual climate variables
        temperature = np.random.normal(15, 5, (10, 50, 50))  # (time, y, x)
        precipitation = np.random.exponential(2, (10, 50, 50))  # (time, y, x)

        # Create coordinates for the output arrays
        x_coords = np.linspace(bbox[0], bbox[2], 50)
        y_coords = np.linspace(bbox[3], bbox[1], 50)
        if time_range is not None:
            time_coords = np.array(pd.date_range(time_range[0], time_range[1], periods=10))
        else:
            time_coords = np.array(pd.date_range("2022-01-01", "2022-12-31", periods=10))

        # Create xarray DataArrays for climate variables
        temperature_da = xr.DataArray(
            temperature,
            dims=("time", "y", "x"),
            coords={"time": time_coords, "y": y_coords, "x": x_coords},
            name="temperature",
            attrs={
                "long_name": "Air Temperature",
                "units": "degrees_celsius",
            },
        )

        precipitation_da = xr.DataArray(
            precipitation,
            dims=("time", "y", "x"),
            coords={"time": time_coords, "y": y_coords, "x": x_coords},
            name="precipitation",
            attrs={
                "long_name": "Precipitation",
                "units": "mm/day",
            },
        )

        return {
            "temperature": temperature_da,
            "precipitation": precipitation_da,
        }

    def extract_all_features(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Optional[Tuple[str, str]] = None,
        resolution: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, xr.DataArray]:
        """Extract all available environmental features for a bounding box and time range.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat).
            time_range: Time range as (start_date, end_date) in ISO format.
                        If None, will use all available times.
            resolution: Output resolution (x_res, y_res) in target CRS units.
                        If None, will use the native resolution of each data source.

        Returns:
            Dictionary of all available features as xarray DataArrays.
        """
        logger.info(f"Extracting all environmental features for bbox {bbox} and time range {time_range}")

        features = {}

        # Extract elevation features if DEM is available
        if self.dem_source is not None:
            elevation_features = self.extract_elevation_features(bbox, resolution)
            features.update(elevation_features)

        # Extract soil features if soil data is available
        if self.soil_source is not None:
            soil_features = self.extract_soil_features(bbox, resolution)
            features.update(soil_features)

        # Extract climate features if climate data is available
        if self.climate_source is not None:
            climate_features = self.extract_climate_features(bbox, time_range)
            features.update(climate_features)

        return features


def zonal_statistics(
    raster_data: Union[np.ndarray, xr.DataArray],
    geometries: gpd.GeoDataFrame,
    transform: Optional[rasterio.transform.Affine] = None,
    stats: List[str] = ["mean", "min", "max", "std"],
) -> gpd.GeoDataFrame:
    """Calculate zonal statistics for geometries.

    Args:
        raster_data: Raster data as numpy array or xarray DataArray.
        geometries: GeoDataFrame containing geometries.
        transform: Affine transform for the raster data.
                   Required if raster_data is a numpy array.
        stats: List of statistics to calculate.

    Returns:
        GeoDataFrame with calculated statistics.
    """
    logger.info(f"Calculating zonal statistics for {len(geometries)} geometries")

    # If raster_data is an xarray DataArray, extract the data and transform
    if isinstance(raster_data, xr.DataArray):
        # This is a simplified implementation
        # In a real application, this would extract the transform from the DataArray's attributes
        # or calculate it from the coordinates
        data = raster_data.values
        if transform is None:
            # Calculate transform from coordinates (simplified)
            x_coords = raster_data.x.values
            y_coords = raster_data.y.values
            pixel_width = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
            pixel_height = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)
            transform = rasterio.transform.from_origin(
                x_coords[0] - pixel_width / 2,
                y_coords[0] - pixel_height / 2,
                pixel_width,
                pixel_height,
            )
    else:
        data = raster_data
        if transform is None:
            raise ValueError("Transform must be provided when raster_data is a numpy array")

    # Create a copy of the GeoDataFrame to store results
    result_gdf = geometries.copy()

    # Calculate statistics for each geometry
    for i, geom in enumerate(geometries.geometry):
        # Create a mask for the geometry
        mask = geometry_mask(
            [mapping(geom)],
            out_shape=data.shape,
            transform=transform,
            invert=True,
        )

        # Extract masked data
        masked_data = data[mask]

        # Calculate statistics
        if len(masked_data) > 0:
            if "mean" in stats:
                result_gdf.loc[i, "mean"] = masked_data.mean()
            if "min" in stats:
                result_gdf.loc[i, "min"] = masked_data.min()
            if "max" in stats:
                result_gdf.loc[i, "max"] = masked_data.max()
            if "std" in stats:
                result_gdf.loc[i, "std"] = masked_data.std()
            if "count" in stats:
                result_gdf.loc[i, "count"] = len(masked_data)
        else:
            # No data points within the geometry
            for stat in stats:
                result_gdf.loc[i, stat] = np.nan

    return result_gdf