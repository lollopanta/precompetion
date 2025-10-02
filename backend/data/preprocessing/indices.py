"""Vegetation indices calculation module.

This module provides functions for calculating various vegetation indices from satellite imagery.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def calculate_ndvi(
    red: Union[np.ndarray, xr.DataArray],
    nir: Union[np.ndarray, xr.DataArray],
    output_name: str = "ndvi",
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Red band reflectance (typically Landsat band 4, Sentinel-2 band 4).
        nir: Near-infrared band reflectance (typically Landsat band 5, Sentinel-2 band 8).
        output_name: Name for the output array if using xarray.

    Returns:
        NDVI values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating NDVI")

    # Handle potential division by zero
    denominator = nir + red
    ndvi = np.where(
        denominator > 0,
        (nir - red) / denominator,
        0,  # Set to 0 where denominator is 0
    )

    # Clip values to valid NDVI range [-1, 1]
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # If input is xarray, return as xarray with metadata
    if isinstance(nir, xr.DataArray):
        return xr.DataArray(
            ndvi,
            dims=nir.dims,
            coords=nir.coords,
            name=output_name,
            attrs={
                "long_name": "Normalized Difference Vegetation Index",
                "units": "-",
                "valid_range": [-1.0, 1.0],
            },
        )
    else:
        return ndvi


def calculate_evi(
    red: Union[np.ndarray, xr.DataArray],
    nir: Union[np.ndarray, xr.DataArray],
    blue: Union[np.ndarray, xr.DataArray],
    output_name: str = "evi",
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate Enhanced Vegetation Index (EVI).

    EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)

    Args:
        red: Red band reflectance.
        nir: Near-infrared band reflectance.
        blue: Blue band reflectance.
        output_name: Name for the output array if using xarray.
        G: Gain factor.
        C1: Coefficient for atmospheric resistance (red band).
        C2: Coefficient for atmospheric resistance (blue band).
        L: Canopy background adjustment.

    Returns:
        EVI values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating EVI")

    # Handle potential division by zero
    denominator = nir + C1 * red - C2 * blue + L
    evi = np.where(
        denominator > 0,
        G * (nir - red) / denominator,
        0,  # Set to 0 where denominator is 0
    )

    # Clip values to reasonable EVI range
    evi = np.clip(evi, -1.0, 1.0)

    # If input is xarray, return as xarray with metadata
    if isinstance(nir, xr.DataArray):
        return xr.DataArray(
            evi,
            dims=nir.dims,
            coords=nir.coords,
            name=output_name,
            attrs={
                "long_name": "Enhanced Vegetation Index",
                "units": "-",
                "valid_range": [-1.0, 1.0],
                "G": G,
                "C1": C1,
                "C2": C2,
                "L": L,
            },
        )
    else:
        return evi


def calculate_savi(
    red: Union[np.ndarray, xr.DataArray],
    nir: Union[np.ndarray, xr.DataArray],
    output_name: str = "savi",
    L: float = 0.5,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate Soil-Adjusted Vegetation Index (SAVI).

    SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)

    Args:
        red: Red band reflectance.
        nir: Near-infrared band reflectance.
        output_name: Name for the output array if using xarray.
        L: Soil brightness correction factor.

    Returns:
        SAVI values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating SAVI")

    # Handle potential division by zero
    denominator = nir + red + L
    savi = np.where(
        denominator > 0,
        (nir - red) / denominator * (1 + L),
        0,  # Set to 0 where denominator is 0
    )

    # Clip values to reasonable SAVI range
    savi = np.clip(savi, -1.0, 1.0)

    # If input is xarray, return as xarray with metadata
    if isinstance(nir, xr.DataArray):
        return xr.DataArray(
            savi,
            dims=nir.dims,
            coords=nir.coords,
            name=output_name,
            attrs={
                "long_name": "Soil-Adjusted Vegetation Index",
                "units": "-",
                "valid_range": [-1.0, 1.0],
                "L": L,
            },
        )
    else:
        return savi


def calculate_ndmi(
    nir: Union[np.ndarray, xr.DataArray],
    swir1: Union[np.ndarray, xr.DataArray],
    output_name: str = "ndmi",
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate Normalized Difference Moisture Index (NDMI).

    NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Args:
        nir: Near-infrared band reflectance.
        swir1: Shortwave infrared band reflectance (typically Landsat band 6).
        output_name: Name for the output array if using xarray.

    Returns:
        NDMI values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating NDMI")

    # Handle potential division by zero
    denominator = nir + swir1
    ndmi = np.where(
        denominator > 0,
        (nir - swir1) / denominator,
        0,  # Set to 0 where denominator is 0
    )

    # Clip values to valid range [-1, 1]
    ndmi = np.clip(ndmi, -1.0, 1.0)

    # If input is xarray, return as xarray with metadata
    if isinstance(nir, xr.DataArray):
        return xr.DataArray(
            ndmi,
            dims=nir.dims,
            coords=nir.coords,
            name=output_name,
            attrs={
                "long_name": "Normalized Difference Moisture Index",
                "units": "-",
                "valid_range": [-1.0, 1.0],
            },
        )
    else:
        return ndmi


def calculate_bloom_index(
    ndvi: Union[np.ndarray, xr.DataArray],
    evi: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ndmi: Optional[Union[np.ndarray, xr.DataArray]] = None,
    output_name: str = "bloom_index",
    weights: Dict[str, float] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate a custom bloom index based on multiple vegetation indices.

    This is a specialized index designed for detecting plant blooming events.
    It combines multiple vegetation indices with optional weighting.

    Args:
        ndvi: Normalized Difference Vegetation Index.
        evi: Enhanced Vegetation Index (optional).
        ndmi: Normalized Difference Moisture Index (optional).
        output_name: Name for the output array if using xarray.
        weights: Dictionary of weights for each index. Defaults to equal weights.

    Returns:
        Bloom index values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating bloom index")

    # Set default weights if not provided
    if weights is None:
        weights = {"ndvi": 1.0}
        if evi is not None:
            weights["evi"] = 1.0
        if ndmi is not None:
            weights["ndmi"] = 0.5

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}

    # Initialize with NDVI (required)
    bloom_index = norm_weights["ndvi"] * ndvi

    # Add EVI if provided
    if evi is not None and "evi" in norm_weights:
        bloom_index += norm_weights["evi"] * evi

    # Add NDMI if provided
    if ndmi is not None and "ndmi" in norm_weights:
        bloom_index += norm_weights["ndmi"] * ndmi

    # Clip values to reasonable range
    bloom_index = np.clip(bloom_index, -1.0, 1.0)

    # If input is xarray, return as xarray with metadata
    if isinstance(ndvi, xr.DataArray):
        return xr.DataArray(
            bloom_index,
            dims=ndvi.dims,
            coords=ndvi.coords,
            name=output_name,
            attrs={
                "long_name": "Bloom Index",
                "units": "-",
                "valid_range": [-1.0, 1.0],
                "weights": str(norm_weights),
            },
        )
    else:
        return bloom_index


def calculate_temporal_ndvi_change(
    ndvi_t1: Union[np.ndarray, xr.DataArray],
    ndvi_t2: Union[np.ndarray, xr.DataArray],
    output_name: str = "ndvi_change",
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate temporal change in NDVI between two time points.

    Args:
        ndvi_t1: NDVI at time point 1.
        ndvi_t2: NDVI at time point 2.
        output_name: Name for the output array if using xarray.

    Returns:
        NDVI change values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating temporal NDVI change")

    # Calculate absolute change
    ndvi_change = ndvi_t2 - ndvi_t1

    # If input is xarray, return as xarray with metadata
    if isinstance(ndvi_t1, xr.DataArray):
        return xr.DataArray(
            ndvi_change,
            dims=ndvi_t1.dims,
            coords=ndvi_t1.coords,
            name=output_name,
            attrs={
                "long_name": "NDVI Change",
                "units": "-",
                "description": "Change in NDVI between two time points",
            },
        )
    else:
        return ndvi_change


def calculate_ndvi_anomaly(
    ndvi: Union[np.ndarray, xr.DataArray],
    ndvi_mean: Union[np.ndarray, xr.DataArray],
    ndvi_std: Union[np.ndarray, xr.DataArray],
    output_name: str = "ndvi_anomaly",
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate NDVI anomaly compared to historical mean.

    Anomaly = (NDVI - Mean) / StdDev

    Args:
        ndvi: Current NDVI values.
        ndvi_mean: Historical mean NDVI values.
        ndvi_std: Historical standard deviation of NDVI values.
        output_name: Name for the output array if using xarray.

    Returns:
        NDVI anomaly values as numpy array or xarray DataArray.
    """
    logger.debug("Calculating NDVI anomaly")

    # Handle potential division by zero
    anomaly = np.where(
        ndvi_std > 0,
        (ndvi - ndvi_mean) / ndvi_std,
        0,  # Set to 0 where std is 0
    )

    # If input is xarray, return as xarray with metadata
    if isinstance(ndvi, xr.DataArray):
        return xr.DataArray(
            anomaly,
            dims=ndvi.dims,
            coords=ndvi.coords,
            name=output_name,
            attrs={
                "long_name": "NDVI Anomaly",
                "units": "standard deviations",
                "description": "Standardized anomaly of NDVI compared to historical mean",
            },
        )
    else:
        return anomaly