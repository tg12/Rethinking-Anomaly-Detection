"""
Fixed Training Script for BWGNN

This module provides training and evaluation functions for BWGNN models.
All configuration variables are defined at the module level.
No command-line arguments or environment variable parsing.

Critical fixes implemented:
1. Proper train/eval mode separation
2. Recompute logits in eval mode for validation
3. Fixed threshold selection methodology using AUPRC
4. Proper data leakage prevention
5. Comprehensive evaluation metrics
"""

# Configuration variables
RANDOM_SEED = 42
DEFAULT_TRAIN_RATIO = 0.4
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 50
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LOG_INTERVAL = 10
VALIDATION_TEST_SPLIT_RATIO = 0.67
NUM_THRESHOLD_STEPS = 100
LOG_LEVEL = "INFO"

import logging
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(
        self,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        lr: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        epoch: int = DEFAULT_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        seed: int = RANDOM_SEED,
        save_model: bool = False,
        save_path: str = "./checkpoints",
    ):
        """
        Initialize training configuration.

        Args:
            train_ratio: Ratio of training data
            lr: Learning rate
            weight_decay: L2 regularization weight
            epoch: Maximum number of training epochs
            patience: Early stopping patience
            grad_clip: Gradient clipping threshold, 0 to disable
            log_interval: Logging interval in epochs
            seed: Random seed
            save_model: Whether to save best model
            save_path: Path to save models
        """
        self.train_ratio = train_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.patience = patience
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.seed = seed
        self.save_model = save_model
        self.save_path = save_path


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        labels: True labels of shape (N,)
        probs: Predicted probabilities of shape (N, num_classes) or (N,)

    Returns:
        Dictionary containing accuracy, precision, recall, F1, AUC-ROC, AUC-PR
    """
    # Extract positive class probabilities
    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    # Generate predictions with default threshold 0.5
    preds = (probs_pos > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "binary_f1": f1_score(labels, preds, average="binary", zero_division=0),
    }

    # Add AUC metrics if both classes present
    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels, probs_pos)
        metrics["auprc"] = average_precision_score(labels, probs_pos)
    else:
        metrics["auc_roc"] = 0.0
        metrics["auprc"] = 0.0

    return metrics


def compute_metrics_with_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    """
    Compute metrics with a specific classification threshold.

    Args:
        labels: True labels
        probs: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics including confusion matrix
    """
    # Extract positive class probabilities
    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    # Apply threshold
    preds = (probs_pos > threshold).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "binary_f1": f1_score(labels, preds, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds),
        "threshold": threshold,
    }

    # Add AUC metrics if both classes present
    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels, probs_pos)
        metrics["auprc"] = average_precision_score(labels, probs_pos)
    else:
        metrics["auc_roc"] = 0.0
        metrics["auprc"] = 0.0

    return metrics


def tune_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
    num_thresholds: int = NUM_THRESHOLD_STEPS,
) -> float:
    """
    Find optimal classification threshold on validation set.

    This should only be called on validation data, never test data,
    to prevent data leakage.

    Args:
        labels: True labels
        probs: Predicted probabilities of shape (N, num_classes) or (N,)
        metric: Metric to optimize ('f1', 'precision', 'recall')
        num_thresholds: Number of threshold values to evaluate

    Returns:
        Optimal threshold value
    """
    # Extract positive class probabilities
    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    best_score = 0.0
    best_threshold = 0.5

    # Evaluate different threshold values
    for threshold in np.linspace(0.01, 0.99, num_thresholds):
        preds = (probs_pos > threshold).astype(int)

        # Calculate metric
        if metric == "f1":
            score = f1_score(labels, preds, average="macro", zero_division=0)
        elif metric == "precision":
            score = precision_score(labels, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Update best threshold
        if score > best_score:
            best_score = score
            best_threshold = threshold

    logger.debug(f"Optimal threshold: {best_threshold:.4f} with {metric}={best_score:.4f}")
    return best_threshold


def train_fixed(
    model, adj_matrix, features, labels, config: TrainingConfig
) -> Tuple[float, float, Dict]:
    """
    Training procedure with proper train/eval separation and metric computation.

    Key fixes:
    1. Recompute logits in eval mode for validation (not reusing train mode logits)
    2. Use AUPRC for model selection instead of threshold-optimized F1
    3. Separate threshold tuning from model selection
    4. Proper gradient handling and early stopping

    Args:
        model: BWGNN model instance
        adj_matrix: Graph adjacency matrix
        features: Node features tensor
        labels: Node labels tensor
        config: Training configuration object

    Returns:
        Tuple of (best_macro_f1, best_auc_roc, metrics_dict)
    """
    # Create train/validation/test splits
    index = list(range(len(labels)))

    idx_train, idx_rest, _, y_rest = train_test_split(
        index,
        labels[index],
        stratify=labels[index],
        train_size=config.train_ratio,
        random_state=config.seed,
        shuffle=True,
    )

    idx_valid, idx_test, _, _ = train_test_split(
        idx_rest,
        y_rest,
        stratify=y_rest,
        test_size=VALIDATION_TEST_SPLIT_RATIO,
        random_state=config.seed,
        shuffle=True,
    )  # Create boolean masks
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_valid] = True
    test_mask[idx_test] = True

    logger.info(
        f"Train/Val/Test split: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}"
    )

    # Calculate class weights for imbalanced data
    pos_count = (labels[train_mask] == 1).sum().item()
    neg_count = (labels[train_mask] == 0).sum().item()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    logger.info(f"Positive class weight: {pos_weight:.4f}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Tracking variables
    best_val_auprc = 0.0
    best_metrics = {}
    patience_counter = 0

    # Training loop
    time_start = time.time()

    for epoch in range(config.epoch):
        # Training phase
        model.train()

        # Forward pass in training mode
        logits_train = model(features, adj_matrix)

        # Compute loss on training set only
        loss = F.cross_entropy(
            logits_train[train_mask],
            labels[train_mask],
            weight=torch.tensor([1.0, pos_weight], device=logits_train.device),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping if enabled
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # Validation phase
        model.eval()

        with torch.no_grad():
            # Recompute logits in eval mode (critical fix)
            logits_eval = model(features, adj_matrix)
            probs = F.softmax(logits_eval, dim=1)

            # Compute validation metrics
            val_metrics = compute_metrics(
                labels[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy()
            )

            # Compute test metrics for monitoring
            test_metrics = compute_metrics(
                labels[test_mask].cpu().numpy(), probs[test_mask].cpu().numpy()
            )

        # Model selection based on validation AUPRC
        val_auprc = val_metrics["auprc"]

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_metrics = {"val": val_metrics, "test": test_metrics, "epoch": epoch}
            patience_counter = 0

            # Save best model if enabled
            if config.save_model:
                import os

                os.makedirs(config.save_path, exist_ok=True)
                torch.save(model.state_dict(), f"{config.save_path}/best_model.pt")
                logger.debug(f"Model saved at epoch {epoch}")
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Logging
        if epoch % config.log_interval == 0:
            logger.info(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                f"Val AUPRC: {val_auprc:.4f} (Best: {best_val_auprc:.4f}) | "
                f"Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Test F1: {test_metrics['macro_f1']:.4f}"
            )

    time_end = time.time()
    logger.info(f"Training completed in {time_end - time_start:.2f}s")

    # Final evaluation with tuned threshold
    logger.info("Final evaluation on test set")

    # Load best model if saved
    if config.save_model:
        model.load_state_dict(torch.load(f"{config.save_path}/best_model.pt"))

    model.eval()
    with torch.no_grad():
        logits_final = model(features, adj_matrix)
        probs_final = F.softmax(logits_final, dim=1)

    # Tune threshold on validation set only
    val_probs = probs_final[val_mask].cpu().numpy()
    val_labels = labels[val_mask].cpu().numpy()
    best_threshold = tune_threshold(val_labels, val_probs, metric="f1")

    logger.info(f"Optimal threshold from validation: {best_threshold:.4f}")

    # Apply threshold to test set
    test_probs = probs_final[test_mask].cpu().numpy()
    test_labels = labels[test_mask].cpu().numpy()

    final_metrics = compute_metrics_with_threshold(test_labels, test_probs, best_threshold)

    # Log final results
    logger.info(f"Test Results (epoch {best_metrics['epoch']}):")
    logger.info(f"  Accuracy:  {final_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Precision: {final_metrics['precision'] * 100:.2f}%")
    logger.info(f"  Recall:    {final_metrics['recall'] * 100:.2f}%")
    logger.info(f"  Macro F1:  {final_metrics['macro_f1'] * 100:.2f}%")
    logger.info(f"  AUC-ROC:   {final_metrics['auc_roc'] * 100:.2f}%")
    logger.info(f"  AUC-PR:    {final_metrics['auprc'] * 100:.2f}%")

    return final_metrics["macro_f1"], final_metrics["auc_roc"], final_metrics
