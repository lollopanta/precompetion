"""Spatial modeling module for bloom prediction.

This module provides models and utilities for spatial prediction of bloom events,
incorporating geospatial features and spatial autocorrelation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class SpatialBloomModel:
    """Model for spatial prediction of bloom events."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        include_spatial_features: bool = True,
        n_neighbors: int = 5,
        distance_weight: float = 0.5,
        random_state: int = 42,
    ):
        """Initialize the spatial bloom model.

        Args:
            model_type: Type of model to use ('lightgbm' or 'random_forest').
            include_spatial_features: Whether to include spatial autocorrelation features.
            n_neighbors: Number of neighbors to consider for spatial features.
            distance_weight: Weight for distance decay in spatial features.
            random_state: Random seed for reproducibility.
        """
        self.model_type = model_type
        self.include_spatial_features = include_spatial_features
        self.n_neighbors = n_neighbors
        self.distance_weight = distance_weight
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importances = None
        self.scaler = StandardScaler()

    def _create_spatial_features(
        self, X: np.ndarray, coords: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Create spatial autocorrelation features.

        Args:
            X: Feature array of shape (n_samples, n_features).
            coords: Coordinate array of shape (n_samples, 2) with [lat, lon].
            y: Optional target array for training data.

        Returns:
            Array with additional spatial features.
        """
        if not self.include_spatial_features:
            return X

        # Calculate distances between all points
        distances = cdist(coords, coords, metric="euclidean")

        # For each point, find the n nearest neighbors
        spatial_features = []

        for i in range(len(X)):
            # Get indices of nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i])[1 : self.n_neighbors + 1]

            # Calculate distance weights (inverse distance)
            weights = 1.0 / (distances[i][neighbor_indices] ** self.distance_weight + 1e-8)
            weights = weights / np.sum(weights)  # Normalize weights

            # Calculate weighted average of neighbor features
            if y is not None:
                # For training data, use target values of neighbors
                neighbor_targets = y[neighbor_indices]
                weighted_target = np.sum(neighbor_targets * weights)
                spatial_features.append([weighted_target])
            else:
                # For prediction data, we don't have targets
                # Use a placeholder (will be updated during prediction)
                spatial_features.append([0.0])

        # Combine original features with spatial features
        return np.hstack([X, np.array(spatial_features)])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "SpatialBloomModel":
        """Fit the model to training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            coords: Coordinate array of shape (n_samples, 2) with [lat, lon].
            feature_names: Optional list of feature names.

        Returns:
            Self for method chaining.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create spatial features if enabled
        if self.include_spatial_features:
            X_with_spatial = self._create_spatial_features(X_scaled, coords, y)
            if feature_names:
                self.feature_names = feature_names + ["spatial_autocorrelation"]
            else:
                self.feature_names = [
                    f"feature_{i}" for i in range(X.shape[1])
                ] + ["spatial_autocorrelation"]
        else:
            X_with_spatial = X_scaled
            self.feature_names = (
                feature_names
                if feature_names
                else [f"feature_{i}" for i in range(X.shape[1])]
            )

        # Initialize and train the model
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
            )
        else:  # random_forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=self.random_state,
            )

        # Train the model
        self.model.fit(X_with_spatial, y)

        # Store feature importances
        self.feature_importances = dict(
            zip(self.feature_names, self.model.feature_importances_)
        )

        return self

    def predict(
        self, X: np.ndarray, coords: np.ndarray, neighbor_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make predictions with the model.

        Args:
            X: Feature array of shape (n_samples, n_features).
            coords: Coordinate array of shape (n_samples, 2) with [lat, lon].
            neighbor_values: Optional array of neighbor target values for spatial features.

        Returns:
            Array of predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create spatial features if enabled
        if self.include_spatial_features:
            if neighbor_values is not None:
                # Use provided neighbor values for spatial features
                X_with_spatial = self._create_spatial_features(X_scaled, coords)
                # Update the spatial feature with provided neighbor values
                X_with_spatial[:, -1] = neighbor_values
            else:
                # For initial prediction without neighbor values
                X_with_spatial = self._create_spatial_features(X_scaled, coords)
        else:
            X_with_spatial = X_scaled

        # Make predictions
        return self.model.predict(X_with_spatial)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, coords: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            coords: Coordinate array of shape (n_samples, 2) with [lat, lon].

        Returns:
            Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = self.predict(X, coords)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {"rmse": rmse, "r2": r2}

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model.
        """
        import joblib

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "include_spatial_features": self.include_spatial_features,
            "n_neighbors": self.n_neighbors,
            "distance_weight": self.distance_weight,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "scaler": self.scaler,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> "SpatialBloomModel":
        """Load a model from a file.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded model.
        """
        import joblib

        model_data = joblib.load(path)

        # Create a new instance
        instance = cls(
            model_type=model_data["model_type"],
            include_spatial_features=model_data["include_spatial_features"],
            n_neighbors=model_data["n_neighbors"],
            distance_weight=model_data["distance_weight"],
            random_state=model_data["random_state"],
        )

        # Restore model attributes
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.feature_importances = model_data["feature_importances"]
        instance.scaler = model_data["scaler"]

        logger.info(f"Model loaded from {path}")
        return instance


class GriddedSpatialModel:
    """Model for spatial prediction on a regular grid."""

    def __init__(
        self,
        base_model: SpatialBloomModel,
        resolution: float = 0.01,  # degrees
        interpolation_method: str = "idw",  # inverse distance weighting
        power: float = 2.0,  # power parameter for IDW
    ):
        """Initialize the gridded spatial model.

        Args:
            base_model: Base spatial model for predictions.
            resolution: Grid resolution in degrees.
            interpolation_method: Method for interpolation ('idw' or 'kriging').
            power: Power parameter for IDW interpolation.
        """
        self.base_model = base_model
        self.resolution = resolution
        self.interpolation_method = interpolation_method
        self.power = power

    def create_prediction_grid(
        self, bounds: Tuple[float, float, float, float], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Create a regular grid for prediction.

        Args:
            bounds: Bounding box as (min_lon, min_lat, max_lon, max_lat).
            crs: Coordinate reference system.

        Returns:
            GeoDataFrame with grid cells.
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Create grid coordinates
        lons = np.arange(min_lon, max_lon, self.resolution)
        lats = np.arange(min_lat, max_lat, self.resolution)

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Flatten grids
        points = np.vstack([lon_grid.flatten(), lat_grid.flatten()]).T

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(points[:, 0], points[:, 1]), crs=crs
        )

        # Add grid indices
        gdf["grid_i"] = np.repeat(np.arange(len(lats)), len(lons))
        gdf["grid_j"] = np.tile(np.arange(len(lons)), len(lats))

        return gdf

    def predict_grid(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        bounds: Tuple[float, float, float, float],
        feature_names: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """Predict bloom probability on a regular grid.

        Args:
            features: Feature array for the grid points.
            coords: Coordinate array with [lon, lat] for each feature row.
            bounds: Bounding box as (min_lon, min_lat, max_lon, max_lat).
            feature_names: Optional list of feature names.

        Returns:
            DataArray with predictions on the grid.
        """
        # Create prediction grid
        grid = self.create_prediction_grid(bounds)

        # Extract grid coordinates
        grid_coords = np.vstack([grid.geometry.x, grid.geometry.y]).T

        # For each grid point, find the nearest feature points
        # and interpolate features
        grid_features = np.zeros((len(grid), features.shape[1]))

        for i, point in enumerate(grid_coords):
            # Calculate distances to all feature points
            distances = np.sqrt(
                np.sum((coords - point) ** 2, axis=1)
            )

            # Find k nearest neighbors
            k = min(5, len(distances))  # Use at most 5 neighbors
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]

            # Avoid division by zero
            nearest_distances = np.maximum(nearest_distances, 1e-8)

            # Calculate weights (inverse distance weighting)
            weights = 1.0 / (nearest_distances ** self.power)
            weights = weights / np.sum(weights)  # Normalize weights

            # Interpolate features
            for j in range(features.shape[1]):
                grid_features[i, j] = np.sum(
                    features[nearest_indices, j] * weights
                )

        # Make predictions on the grid
        grid_predictions = self.base_model.predict(
            grid_features, grid_coords
        )

        # Reshape predictions to 2D grid
        min_lon, min_lat, max_lon, max_lat = bounds
        lons = np.arange(min_lon, max_lon, self.resolution)
        lats = np.arange(min_lat, max_lat, self.resolution)

        # Create DataArray
        prediction_grid = np.zeros((len(lats), len(lons)))
        for i, row in grid.iterrows():
            prediction_grid[int(row["grid_i"]), int(row["grid_j"])] = grid_predictions[i]

        # Create xarray DataArray
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
                "resolution": self.resolution,
            },
        )

        return da


def train_spatial_bloom_model(
    data: gpd.GeoDataFrame,
    feature_cols: List[str],
    target_col: str,
    coord_cols: List[str] = ["longitude", "latitude"],
    model_type: str = "lightgbm",
    include_spatial: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    model_save_path: Optional[str] = None,
) -> Tuple[SpatialBloomModel, Dict[str, float]]:
    """Train a spatial bloom model.

    Args:
        data: GeoDataFrame with features, target, and coordinates.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        coord_cols: Names of coordinate columns [longitude, latitude].
        model_type: Type of model to use ('lightgbm' or 'random_forest').
        include_spatial: Whether to include spatial autocorrelation features.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        model_save_path: Optional path to save the trained model.

    Returns:
        Tuple of (model, evaluation_metrics).
    """
    # Extract features, target, and coordinates
    X = data[feature_cols].values
    y = data[target_col].values
    coords = data[coord_cols].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=test_size, random_state=random_state
    )

    # Create and train the model
    model = SpatialBloomModel(
        model_type=model_type,
        include_spatial_features=include_spatial,
        random_state=random_state,
    )

    model.fit(X_train, y_train, coords_train, feature_names=feature_cols)

    # Evaluate the model
    metrics = model.evaluate(X_test, y_test, coords_test)
    logger.info(
        f"Model evaluation: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}"
    )

    # Print feature importances
    importances = sorted(
        model.feature_importances.items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Feature importances:")
    for feature, importance in importances:
        logger.info(f"  {feature}: {importance:.4f}")

    # Save the model if a path is provided
    if model_save_path is not None:
        model.save_model(model_save_path)

    return model, metrics