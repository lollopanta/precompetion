"""Bloom prediction inference module.

This module provides utilities for making bloom predictions using trained models.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import torch
import joblib
from shapely.geometry import Point

# Import model classes from training modules
from backend.models.training.bloom_classifier import BloomClassifier, BloomClassifierTrainer
from backend.models.training.bloom_forecaster import BloomForecaster, BloomForecasterTrainer
from backend.models.training.spatial_model import SpatialBloomModel, GriddedSpatialModel

logger = logging.getLogger(__name__)


class BloomPredictor:
    """Main class for bloom prediction using multiple models."""

    def __init__(
        self,
        model_dir: str,
        classifier_model_path: Optional[str] = None,
        forecaster_model_path: Optional[str] = None,
        spatial_model_path: Optional[str] = None,
        feature_scaler_path: Optional[str] = None,
    ):
        """Initialize the bloom predictor.

        Args:
            model_dir: Directory containing model files.
            classifier_model_path: Path to the bloom classifier model.
            forecaster_model_path: Path to the bloom forecaster model.
            spatial_model_path: Path to the spatial bloom model.
            feature_scaler_path: Path to the feature scaler.
        """
        self.model_dir = Path(model_dir)
        self.classifier = None
        self.forecaster = None
        self.spatial_model = None
        self.feature_scaler = None

        # Load models if paths are provided
        if classifier_model_path:
            self.load_classifier(classifier_model_path)

        if forecaster_model_path:
            self.load_forecaster(forecaster_model_path)

        if spatial_model_path:
            self.load_spatial_model(spatial_model_path)

        if feature_scaler_path:
            self.load_feature_scaler(feature_scaler_path)

    def load_classifier(self, model_path: str) -> None:
        """Load the bloom classifier model.

        Args:
            model_path: Path to the model file.
        """
        try:
            trainer, model = BloomClassifierTrainer.load_model(model_path)
            self.classifier = trainer
            logger.info(f"Loaded bloom classifier from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load bloom classifier: {e}")
            raise

    def load_forecaster(self, model_path: str) -> None:
        """Load the bloom forecaster model.

        Args:
            model_path: Path to the model file.
        """
        try:
            trainer, model = BloomForecasterTrainer.load_model(model_path)
            self.forecaster = trainer
            logger.info(f"Loaded bloom forecaster from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load bloom forecaster: {e}")
            raise

    def load_spatial_model(self, model_path: str) -> None:
        """Load the spatial bloom model.

        Args:
            model_path: Path to the model file.
        """
        try:
            self.spatial_model = SpatialBloomModel.load_model(model_path)
            logger.info(f"Loaded spatial bloom model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load spatial bloom model: {e}")
            raise

    def load_feature_scaler(self, scaler_path: str) -> None:
        """Load the feature scaler.

        Args:
            scaler_path: Path to the scaler file.
        """
        try:
            self.feature_scaler = joblib.load(scaler_path)
            logger.info(f"Loaded feature scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to load feature scaler: {e}")
            raise

    def predict_bloom_probability(
        self, features: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict bloom probability using the classifier model.

        Args:
            features: Feature array of shape (n_samples, n_features).
            threshold: Classification threshold.

        Returns:
            Tuple of (probabilities, binary_predictions).
        """
        if self.classifier is None:
            raise ValueError("Classifier model not loaded.")

        # Scale features if scaler is available
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Make predictions
        return self.classifier.predict(features_tensor, threshold=threshold)

    def predict_bloom_forecast(
        self, features: np.ndarray, seq_length: int, forecast_horizon: int
    ) -> np.ndarray:
        """Predict bloom forecast using the forecaster model.

        Args:
            features: Feature array of shape (n_samples, n_features).
            seq_length: Length of input sequence.
            forecast_horizon: Number of future time steps to predict.

        Returns:
            Array of forecasted values.
        """
        if self.forecaster is None:
            raise ValueError("Forecaster model not loaded.")

        # Scale features if scaler is available
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)

        # Reshape for sequence input (batch_size, seq_length, n_features)
        if len(features.shape) == 2:
            # Assume we have a single sequence
            features = features[-seq_length:].reshape(1, seq_length, -1)

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Generate forecast
        with torch.no_grad():
            forecast = self.forecaster.model.predict_sequence(
                features_tensor, forecast_horizon
            )

        return forecast.cpu().numpy()

    def predict_spatial(
        self, features: np.ndarray, coords: np.ndarray
    ) -> np.ndarray:
        """Make spatial predictions using the spatial model.

        Args:
            features: Feature array of shape (n_samples, n_features).
            coords: Coordinate array of shape (n_samples, 2) with [lat, lon].

        Returns:
            Array of predictions.
        """
        if self.spatial_model is None:
            raise ValueError("Spatial model not loaded.")

        return self.spatial_model.predict(features, coords)

    def predict_grid(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        bounds: Tuple[float, float, float, float],
        resolution: float = 0.01,
    ) -> xr.DataArray:
        """Predict bloom probability on a regular grid.

        Args:
            features: Feature array for reference points.
            coords: Coordinate array with [lon, lat] for reference points.
            bounds: Bounding box as (min_lon, min_lat, max_lon, max_lat).
            resolution: Grid resolution in degrees.

        Returns:
            DataArray with predictions on the grid.
        """
        if self.spatial_model is None:
            raise ValueError("Spatial model not loaded.")

        # Create gridded model
        gridded_model = GriddedSpatialModel(
            base_model=self.spatial_model, resolution=resolution
        )

        # Make grid predictions
        return gridded_model.predict_grid(features, coords, bounds)

    def predict_location(
        self,
        lat: float,
        lon: float,
        date: Union[str, datetime],
        environmental_features: Dict[str, float],
    ) -> Dict[str, Union[float, str]]:
        """Predict bloom metrics for a specific location and date.

        Args:
            lat: Latitude.
            lon: Longitude.
            date: Date for prediction.
            environmental_features: Dictionary of environmental features.

        Returns:
            Dictionary with prediction results.
        """
        # Convert date to datetime if it's a string
        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        # Extract features from environmental data
        feature_values = list(environmental_features.values())
        feature_array = np.array(feature_values).reshape(1, -1)

        # Coordinates
        coords = np.array([[lon, lat]])

        # Make predictions
        results = {}

        # Bloom probability (current)
        if self.classifier is not None:
            probs, binary = self.predict_bloom_probability(feature_array)
            results["bloom_probability"] = float(probs[0])
            results["is_blooming"] = bool(binary[0])

        # Spatial context
        if self.spatial_model is not None:
            spatial_pred = self.predict_spatial(feature_array, coords)
            results["spatial_bloom_intensity"] = float(spatial_pred[0])

        # Time forecast (if we have time series data)
        if self.forecaster is not None:
            # This would require historical data for this location
            # For simplicity, we'll just note that forecasting requires time series
            results["forecast_available"] = False
            results["forecast_message"] = "Time series forecasting requires historical data"

        # Add metadata
        results["latitude"] = lat
        results["longitude"] = lon
        results["prediction_date"] = date.isoformat()
        results["prediction_timestamp"] = datetime.now().isoformat()

        return results


def load_predictor(model_dir: str) -> BloomPredictor:
    """Load a bloom predictor with all available models.

    Args:
        model_dir: Directory containing model files.

    Returns:
        Loaded BloomPredictor instance.
    """
    model_dir_path = Path(model_dir)

    # Find model files
    classifier_path = None
    forecaster_path = None
    spatial_path = None
    scaler_path = None

    # Look for model files with standard names
    if (model_dir_path / "bloom_classifier.pt").exists():
        classifier_path = str(model_dir_path / "bloom_classifier.pt")

    if (model_dir_path / "bloom_forecaster.pt").exists():
        forecaster_path = str(model_dir_path / "bloom_forecaster.pt")

    if (model_dir_path / "spatial_model.joblib").exists():
        spatial_path = str(model_dir_path / "spatial_model.joblib")

    if (model_dir_path / "feature_scaler.joblib").exists():
        scaler_path = str(model_dir_path / "feature_scaler.joblib")

    # Create and return predictor
    return BloomPredictor(
        model_dir=model_dir,
        classifier_model_path=classifier_path,
        forecaster_model_path=forecaster_path,
        spatial_model_path=spatial_path,
        feature_scaler_path=scaler_path,
    )