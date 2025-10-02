"""Bloom classification model training module.

This module provides a PyTorch-based model for classifying bloom events from satellite imagery.
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BloomDataset(Dataset):
    """Dataset for bloom classification."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None,
    ):
        """Initialize the bloom dataset.

        Args:
            features: Feature array of shape (n_samples, n_features).
            labels: Label array of shape (n_samples,).
            transform: Optional transform to apply to the features.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (features, label).
        """
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label


class BloomClassifier(nn.Module):
    """Neural network model for bloom classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.3,
    ):
        """Initialize the bloom classifier.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate for regularization.
        """
        super(BloomClassifier, self).__init__()

        # Build the network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        return self.model(x)


class BloomClassifierTrainer:
    """Trainer for the bloom classifier model."""

    def __init__(
        self,
        model: BloomClassifier,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        """Initialize the trainer.

        Args:
            model: The bloom classifier model.
            learning_rate: Learning rate for optimization.
            weight_decay: Weight decay for regularization.
        """
        self.model = model
        self.criterion = nn.BCELoss()
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
        logger.info(f"Training bloom classifier for {num_epochs} epochs")
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auc": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels.view(-1, 1))

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * features.size(0)

            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])
                history["val_auc"].append(val_metrics["auc"])

                # Print progress
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f}, "
                        f"Val AUC: {val_metrics['auc']:.4f}"
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
        self, data_loader: DataLoader, threshold: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation data.
            threshold: Classification threshold.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels.view(-1, 1))

                total_loss += loss.item() * features.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)

        # Convert predictions to binary using threshold
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)
        binary_preds = (all_preds >= threshold).astype(int)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average="binary"
        )
        auc = roc_auc_score(all_labels, all_preds)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

        return avg_loss, metrics

    def predict(
        self, features: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the model.

        Args:
            features: Input features tensor.
            threshold: Classification threshold.

        Returns:
            Tuple of (probabilities, binary_predictions).
        """
        self.model.eval()
        features = features.to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            probs = outputs.cpu().numpy().flatten()
            preds = (probs >= threshold).astype(int)

        return probs, preds

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.model[0].in_features,
                "hidden_dims": [
                    layer.out_features
                    for layer in self.model.model
                    if isinstance(layer, nn.Linear)
                ][:-1],  # Exclude output layer
                "dropout_rate": self.model.model[3].p,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> Tuple["BloomClassifierTrainer", BloomClassifier]:
        """Load a model from a file.

        Args:
            path: Path to the saved model.

        Returns:
            Tuple of (trainer, model).
        """
        checkpoint = torch.load(path)
        model = BloomClassifier(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = cls(model)
        logger.info(f"Model loaded from {path}")
        return trainer, model


def prepare_bloom_data(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Prepare data for bloom classification.

    Args:
        features_df: DataFrame containing features.
        labels_df: DataFrame containing labels.
        feature_cols: List of feature column names.
        label_col: Name of the label column.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of data to use for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (data_dict, feature_names).
    """
    # Extract features and labels
    X = features_df[feature_cols].values
    y = labels_df[label_col].values

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )

    # Return data dictionary and feature names
    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return data_dict, feature_cols


def train_bloom_classifier(
    data_dict: Dict[str, np.ndarray],
    input_dim: int,
    hidden_dims: List[int] = [128, 64],
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    model_save_path: Optional[str] = None,
) -> Tuple[BloomClassifierTrainer, Dict[str, List[float]]]:
    """Train a bloom classifier model.

    Args:
        data_dict: Dictionary containing training, validation, and test data.
        input_dim: Number of input features.
        hidden_dims: List of hidden layer dimensions.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization.
        model_save_path: Optional path to save the trained model.

    Returns:
        Tuple of (trainer, history).
    """
    # Create datasets and data loaders
    train_dataset = BloomDataset(data_dict["X_train"], data_dict["y_train"])
    val_dataset = BloomDataset(data_dict["X_val"], data_dict["y_val"])
    test_dataset = BloomDataset(data_dict["X_test"], data_dict["y_test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create and train the model
    model = BloomClassifier(input_dim=input_dim, hidden_dims=hidden_dims)
    trainer = BloomClassifierTrainer(
        model, learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Train the model
    history = trainer.train(
        train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs
    )

    # Evaluate on test set
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Test Loss: {test_loss:.4f}, "
        f"Test F1: {test_metrics['f1']:.4f}, "
        f"Test AUC: {test_metrics['auc']:.4f}"
    )

    # Save the model if a path is provided
    if model_save_path is not None:
        trainer.save_model(model_save_path)

    return trainer, history