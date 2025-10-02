"""Bloom forecasting model training module.

This module provides a PyTorch-based model for forecasting bloom events over time.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int,
        forecast_horizon: int,
    ):
        """Initialize the time series dataset.

        Args:
            features: Feature array of shape (n_samples, n_features).
            targets: Target array of shape (n_samples,).
            seq_length: Length of input sequence.
            forecast_horizon: Number of future time steps to predict.
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.indices = self._create_indices()

    def _create_indices(self) -> List[Tuple[int, int]]:
        """Create valid (start, end) index pairs for sequences.

        Returns:
            List of (start_idx, end_idx) tuples.
        """
        indices = []
        for i in range(len(self.features) - self.seq_length - self.forecast_horizon + 1):
            indices.append((i, i + self.seq_length))
        return indices

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence from the dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Tuple of (input_sequence, target_sequence).
        """
        start_idx, end_idx = self.indices[idx]
        
        # Input sequence
        input_seq = self.features[start_idx:end_idx]
        
        # Target sequence (future values to predict)
        target_seq = self.targets[end_idx:end_idx + self.forecast_horizon]
        
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(target_seq, dtype=torch.float32)
        )


class BloomForecaster(nn.Module):
    """LSTM-based model for bloom forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        """Initialize the bloom forecaster.

        Args:
            input_dim: Number of input features.
            hidden_dim: Hidden dimension of the LSTM.
            num_layers: Number of LSTM layers.
            output_dim: Dimension of the output (typically 1 for bloom intensity).
            dropout: Dropout rate for regularization.
        """
        super(BloomForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim).
            hidden: Optional initial hidden state.

        Returns:
            Tuple of (output, (hidden_state, cell_state)).
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply output layer to the last time step
        output = self.fc(lstm_out[:, -1, :])
        
        return output, hidden

    def predict_sequence(
        self, x: torch.Tensor, forecast_length: int
    ) -> torch.Tensor:
        """Generate a sequence of predictions.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim).
            forecast_length: Number of future steps to predict.

        Returns:
            Tensor of predictions of shape (batch_size, forecast_length).
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Initialize output tensor
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, forecast_length, device=device)
        
        # Initialize hidden state
        hidden = None
        
        # Generate predictions one step at a time
        current_input = x
        
        for t in range(forecast_length):
            # Get prediction for current step
            with torch.no_grad():
                output, hidden = self(current_input, hidden)
            
            # Store prediction
            outputs[:, t] = output.squeeze()
            
            # Update input for next step (use prediction as new feature)
            # This is a simplified approach; in practice, you might need to transform
            # the prediction to match all input features
            last_step = current_input[:, -1, :].clone()
            last_step[:, 0] = output.squeeze()  # Assuming first feature is the target
            
            # Remove first time step and append new prediction
            current_input = torch.cat([
                current_input[:, 1:, :],
                last_step.unsqueeze(1)
            ], dim=1)
        
        return outputs


class BloomForecasterTrainer:
    """Trainer for the bloom forecaster model."""

    def __init__(
        self,
        model: BloomForecaster,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        """Initialize the trainer.

        Args:
            model: The bloom forecaster model.
            learning_rate: Learning rate for optimization.
            weight_decay: Weight decay for regularization.
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            num_epochs: Number of training epochs.
            patience: Number of epochs to wait for improvement before early stopping.
            verbose: Whether to print training progress.

        Returns:
            Dictionary of training history (loss and metrics).
        """
        logger.info(f"Training bloom forecaster for {num_epochs} epochs")
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_r2": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_rmse"].append(val_metrics["rmse"])
                history["val_mae"].append(val_metrics["mae"])
                history["val_r2"].append(val_metrics["r2"])

                # Print progress
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val RMSE: {val_metrics['rmse']:.4f}"
                    )

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # Print progress without validation
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}"
                    )

        # Restore best model if validation was used
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history

    def evaluate(
        self, data_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation data.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        return avg_loss, metrics

    def predict(
        self, inputs: torch.Tensor
    ) -> np.ndarray:
        """Make predictions with the model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            Numpy array of predictions.
        """
        self.model.eval()
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs, _ = self.model(inputs)
            predictions = outputs.cpu().numpy()

        return predictions

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "output_dim": self.model.output_dim,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> Tuple["BloomForecasterTrainer", BloomForecaster]:
        """Load a model from a file.

        Args:
            path: Path to the saved model.

        Returns:
            Tuple of (trainer, model).
        """
        checkpoint = torch.load(path)
        model = BloomForecaster(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            output_dim=checkpoint["output_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = cls(model)
        logger.info(f"Model loaded from {path}")
        return trainer, model


def prepare_time_series_data(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_length: int = 12,  # e.g., 12 months of history
    forecast_horizon: int = 3,  # e.g., predict next 3 months
    test_size: float = 0.2,
    val_size: float = 0.1,
    scale_features: bool = True,
    random_state: int = 42,
) -> Tuple[Dict[str, Union[np.ndarray, DataLoader]], Dict[str, StandardScaler]]:
    """Prepare time series data for bloom forecasting.

    Args:
        data: DataFrame containing time series data.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        seq_length: Length of input sequence.
        forecast_horizon: Number of future time steps to predict.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of data to use for validation.
        scale_features: Whether to scale features.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (data_dict, scalers_dict).
    """
    # Extract features and target
    features = data[feature_cols].values
    targets = data[[target_col]].values
    
    # Initialize scalers
    scalers = {}
    
    # Scale features if requested
    if scale_features:
        feature_scaler = StandardScaler()
        features = feature_scaler.fit_transform(features)
        scalers["feature_scaler"] = feature_scaler
        
        target_scaler = StandardScaler()
        targets = target_scaler.fit_transform(targets)
        scalers["target_scaler"] = target_scaler
    
    # Split data into train, validation, and test sets
    # For time series, we need to maintain temporal order
    train_size = 1 - test_size - val_size
    train_end = int(len(features) * train_size)
    val_end = int(len(features) * (train_size + val_size))
    
    X_train = features[:train_end]
    y_train = targets[:train_end]
    
    X_val = features[train_end:val_end]
    y_val = targets[train_end:val_end]
    
    X_test = features[val_end:]
    y_test = targets[val_end:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length, forecast_horizon)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_length, forecast_horizon)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length, forecast_horizon)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Return data dictionary and scalers
    data_dict = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    
    return data_dict, scalers


def train_bloom_forecaster(
    data_dict: Dict[str, Union[np.ndarray, DataLoader]],
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    output_dim: int = 1,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    model_save_path: Optional[str] = None,
) -> Tuple[BloomForecasterTrainer, Dict[str, List[float]]]:
    """Train a bloom forecaster model.

    Args:
        data_dict: Dictionary containing training, validation, and test data.
        input_dim: Number of input features.
        hidden_dim: Hidden dimension of the LSTM.
        num_layers: Number of LSTM layers.
        output_dim: Dimension of the output.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization.
        model_save_path: Optional path to save the trained model.

    Returns:
        Tuple of (trainer, history).
    """
    # Create the model
    model = BloomForecaster(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
    )
    
    # Create the trainer
    trainer = BloomForecasterTrainer(
        model, learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    # Train the model
    history = trainer.train(
        train_loader=data_dict["train_loader"],
        val_loader=data_dict["val_loader"],
        num_epochs=num_epochs,
    )
    
    # Evaluate on test set
    test_loss, test_metrics = trainer.evaluate(data_dict["test_loader"])
    logger.info(
        f"Test Loss: {test_loss:.4f}, "
        f"Test RMSE: {test_metrics['rmse']:.4f}, "
        f"Test R²: {test_metrics['r2']:.4f}"
    )
    
    # Save the model if a path is provided
    if model_save_path is not None:
        trainer.save_model(model_save_path)
    
    return trainer, history