"""
Fraud Detection Demo using BWGNN.

Demonstrates spectral graph convolution with Beta wavelets for detecting
fraudulent transactions in a synthetic financial network.
"""

import logging
import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn

# Configuration variables
RANDOM_SEED = 42
NUM_NODES = 300
NUM_NORMAL_COMMUNITIES = 5
FRAUD_RATIO = 0.15
FEATURE_DIM = 16
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
EPOCHS = 150
TRAIN_RATIO = 0.4
BETA_WAVELET_ORDER = 3
DROPOUT_RATE = 0.5
LOG_LEVEL = logging.INFO
ENABLE_VISUALIZATION = True

# Configure structured logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# Set random seeds for deterministic behavior
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for visualization support
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, visualization disabled")


def calculate_beta_wavelet_coefficients(eigenvalues: torch.Tensor, d: int = 2) -> torch.Tensor:
    """
    Calculate Beta wavelet coefficients for polynomial approximation.

    Beta wavelets provide a multi-scale representation of graph signals by
    applying scaling and wavelet functions at different scales.

    Args:
        eigenvalues: Eigenvalues of normalized Laplacian in range [0, 2]
        d: Order of polynomial approximation

    Returns:
        Tensor of shape [d+1, num_eigenvalues] containing wavelet coefficients
    """
    logger.debug(f"Calculating Beta wavelet coefficients: d={d}, eigenvalues={len(eigenvalues)}")

    def scaling_function(x: torch.Tensor) -> torch.Tensor:
        """Low-pass filter: passes frequencies <= 1."""
        return torch.where(x <= 1, torch.ones_like(x), torch.zeros_like(x))

    def wavelet_function(x: torch.Tensor) -> torch.Tensor:
        """Band-pass filter: positive in [0.5, 1], negative in [1, 2]."""
        return torch.where(
            (x > 0.5) & (x <= 1),
            torch.ones_like(x),
            torch.where((x > 1) & (x < 2), -torch.ones_like(x), torch.zeros_like(x)),
        )

    # Apply wavelets at different scales for multi-resolution analysis
    theta = []
    for k in range(d + 1):
        scale = 2**k
        psi = wavelet_function(scale * eigenvalues / 2)
        theta.append(psi)

    return torch.stack(theta, dim=0)


class SimpleBWGNN(nn.Module):
    """
    Simplified BWGNN model for fraud detection.

    Uses spectral graph convolution with Beta wavelets to detect
    anomalous patterns in transaction networks.
    """

    def __init__(
        self,
        in_feats: int,
        h_feats: int,
        num_classes: int,
        adj_matrix: sp.coo_matrix,
        d: int = 3,
    ):
        """
        Initialize BWGNN model.

        Args:
            in_feats: Input feature dimension
            h_feats: Hidden layer dimension
            num_classes: Number of output classes
            adj_matrix: Sparse adjacency matrix
            d: Order of Beta wavelet polynomial
        """
        super(SimpleBWGNN, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.d = d

        logger.info(
            f"Initializing BWGNN: in={in_feats}, hidden={h_feats}, "
            f"classes={num_classes}, wavelet_order={d}"
        )

        # Store adjacency matrix as dense tensor
        self.register_buffer("adj", torch.FloatTensor(adj_matrix.toarray()))

        # Precompute spectral decomposition
        self._precompute_spectral()

        # Define network layers
        self.conv1 = nn.Linear(in_feats, h_feats)
        self.conv2 = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def _precompute_spectral(self) -> None:
        """
        Precompute eigenvalues and eigenvectors of normalized Laplacian.

        Computes L = I - D^(-1/2) A D^(-1/2) where:
        - I is identity matrix
        - D is degree matrix
        - A is adjacency matrix
        """
        logger.debug("Computing spectral decomposition of graph Laplacian")

        adj = self.adj.numpy()
        n = adj.shape[0]

        # Compute degree matrix
        degrees = adj.sum(axis=1)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))

        # Compute normalized adjacency: D^(-1/2) A D^(-1/2)
        A_norm = D_sqrt_inv @ adj @ D_sqrt_inv

        # Compute normalized Laplacian: L = I - A_norm
        L = np.eye(n) - A_norm

        # Eigendecomposition of Laplacian
        eigenvalues, eigenvectors = eigh(L)

        # Store as buffers for gradient-free access
        self.register_buffer("eigenvalues", torch.FloatTensor(eigenvalues))
        self.register_buffer("eigenvectors", torch.FloatTensor(eigenvectors))

        # Calculate Beta wavelet coefficients
        theta = calculate_beta_wavelet_coefficients(self.eigenvalues, d=self.d)
        self.register_buffer("theta", theta)

        logger.debug(f"Spectral decomposition complete: {len(eigenvalues)} eigenvalues")

    def spectral_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Beta wavelet spectral filtering.

        Transforms features to spectral domain, applies wavelet filters,
        and transforms back to spatial domain.

        Args:
            x: Input features of shape [num_nodes, num_features]

        Returns:
            Filtered features of shape [num_nodes, num_features]
        """
        # Transform to spectral domain using eigenvectors
        x_spectral = self.eigenvectors.T @ x

        # Apply Beta wavelet filtering across scales
        filtered = torch.zeros_like(x_spectral)
        for k in range(self.d + 1):
            # Apply k-th wavelet coefficient (element-wise scaling)
            filtered += self.theta[k].unsqueeze(-1) * x_spectral

        # Transform back to spatial domain
        x_filtered = self.eigenvectors @ filtered

        return x_filtered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spectral convolution.

        Args:
            x: Input features of shape [num_nodes, in_feats]

        Returns:
            Logits of shape [num_nodes, num_classes]
        """
        # First layer: spectral filtering + linear transform + activation
        x = self.spectral_filter(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second layer: spectral filtering + linear transform
        x = self.spectral_filter(x)
        x = self.conv2(x)

        return x


def create_fraud_network() -> Tuple[sp.coo_matrix, torch.Tensor, torch.Tensor]:
    """
    Create synthetic transaction network with fraud patterns.

    Normal users form tight communities (friends paying friends).
    Fraudsters connect randomly to victims (anomalous pattern).

    Returns:
        adj_matrix: Sparse adjacency matrix
        features: Node features tensor
        labels: Binary labels (0=normal, 1=fraud)
    """
    logger.info("Creating synthetic financial network")

    community_size = NUM_NODES // NUM_NORMAL_COMMUNITIES
    edges = []
    features = []
    labels = []

    # Generate normal communities with dense intra-community connections
    for comm_id in range(NUM_NORMAL_COMMUNITIES):
        start_node = comm_id * community_size
        end_node = start_node + community_size

        for i in range(start_node, end_node):
            # Community-specific feature distribution
            base_feature = np.random.randn(FEATURE_DIM) + comm_id * 2
            features.append(base_feature)
            labels.append(0)

            # Connect to random nodes within same community
            num_connections = np.random.randint(5, 11)
            targets = np.random.choice(
                range(start_node, end_node),
                size=min(num_connections, end_node - start_node - 1),
                replace=False,
            )
            for target in targets:
                if target != i:
                    edges.append((i, target))
                    edges.append((target, i))

    # Add sparse cross-community edges
    num_cross_edges = NUM_NODES // 2
    for _ in range(num_cross_edges):
        src = np.random.randint(0, NUM_NODES)
        dst = np.random.randint(0, NUM_NODES)
        if src != dst:
            edges.append((src, dst))
            edges.append((dst, src))

    # Inject fraudulent nodes with anomalous patterns
    num_fraud = int(NUM_NODES * FRAUD_RATIO)
    fraud_nodes = np.random.choice(NUM_NODES, size=num_fraud, replace=False)

    for fraud_node in fraud_nodes:
        labels[fraud_node] = 1

        # Fraudulent features: different distribution
        features[fraud_node] = np.random.randn(FEATURE_DIM) * 3 + 10

        # Fraudulent connectivity: random victims (not community-based)
        num_victims = np.random.randint(8, 15)
        victims = np.random.choice(NUM_NODES, size=num_victims, replace=False)
        for victim in victims:
            if victim != fraud_node:
                edges.append((fraud_node, victim))
                edges.append((victim, fraud_node))

    # Build sparse adjacency matrix
    edges = list(set(edges))
    rows, cols = zip(*edges) if edges else ([], [])
    data = np.ones(len(rows))
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(NUM_NODES, NUM_NODES))

    # Convert to tensors
    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Normalize features (zero mean, unit variance)
    features = (features - features.mean(0)) / (features.std(0) + 1e-5)

    logger.info(
        f"Network created: {NUM_NODES} nodes, {len(edges)} edges, "
        f"{num_fraud} fraudulent ({FRAUD_RATIO * 100:.1f}%)"
    )

    return adj_matrix, features, labels


def visualize_results(labels: torch.Tensor, predictions: np.ndarray) -> None:
    """
    Create visualization of fraud detection results.

    Generates confusion matrix heatmap and performance metrics bar chart.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels
    """
    if not HAS_MATPLOTLIB or not ENABLE_VISUALIZATION:
        return

    logger.info("Generating visualization")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot confusion matrix
    cm = confusion_matrix(labels.numpy(), predictions)
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Normal", "Fraud"])
    ax1.set_yticklabels(["Normal", "Fraud"])
    ax1.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Add value annotations to confusion matrix
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=color,
                fontsize=20,
                fontweight="bold",
            )

    plt.colorbar(im1, ax=ax1)

    # Plot performance metrics
    from sklearn.metrics import f1_score, precision_score, recall_score

    precision = precision_score(labels.numpy(), predictions)
    recall = recall_score(labels.numpy(), predictions)
    f1 = f1_score(labels.numpy(), predictions)
    accuracy = (predictions == labels.numpy()).mean()

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy, precision, recall, f1]
    colors_bar = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    bars = ax2.bar(metrics, values, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=2)
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_title("Performance Metrics", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("fraud_detection_results.png", dpi=150, bbox_inches="tight")
    logger.info("Visualization saved to fraud_detection_results.png")
    plt.close()


def train_and_evaluate() -> Tuple[SimpleBWGNN, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train BWGNN model and evaluate fraud detection performance.

    Returns:
        model: Trained BWGNN model
        features: Node features
        labels: Ground truth labels
        predictions: Model predictions
    """
    # Create synthetic network
    adj_matrix, features, labels = create_fraud_network()

    # Split into train/validation/test sets with stratification
    indices = list(range(len(labels)))
    idx_train, idx_test = train_test_split(
        indices,
        stratify=labels[indices],
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    idx_train, idx_val = train_test_split(
        idx_train,
        stratify=labels[idx_train],
        test_size=0.25,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # Create boolean masks for splits
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    logger.info(
        f"Dataset split: train={train_mask.sum().item()}, "
        f"val={val_mask.sum().item()}, test={test_mask.sum().item()}"
    )

    # Initialize model
    model = SimpleBWGNN(
        in_feats=FEATURE_DIM,
        h_feats=HIDDEN_DIM,
        num_classes=2,
        adj_matrix=adj_matrix,
        d=BETA_WAVELET_ORDER,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Calculate class weights for imbalanced data
    fraud_weight = (labels[train_mask] == 0).sum().float() / (labels[train_mask] == 1).sum().float()
    class_weights = torch.tensor([1.0, fraud_weight.item()])

    logger.info(
        f"Training configuration: lr={LEARNING_RATE}, fraud_weight={fraud_weight.item():.2f}"
    )

    # Training loop
    logger.info(f"Starting training for {EPOCHS} epochs")
    best_val_auc = 0
    best_epoch = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_probs = F.softmax(val_logits, dim=1)
            val_auc = roc_auc_score(labels[val_mask].numpy(), val_probs[val_mask, 1].numpy())

            # Track best model based on validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Log progress every 30 epochs
        if (epoch + 1) % 30 == 0:
            logger.info(
                f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | "
                f"Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}"
            )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f}s, best epoch: {best_epoch + 1}")

    # Load best model state
    model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(features)
        test_probs = F.softmax(test_logits, dim=1)
        test_preds = test_probs.argmax(dim=1)

        # Calculate test metrics
        test_auc = roc_auc_score(labels[test_mask].numpy(), test_probs[test_mask, 1].numpy())

        logger.info("=" * 70)
        logger.info("FRAUD DETECTION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Test Set AUC: {test_auc:.4f}")
        logger.info("\nClassification Report:")
        print(
            classification_report(
                labels[test_mask].numpy(),
                test_preds[test_mask].numpy(),
                target_names=["Normal", "Fraud"],
                digits=4,
            )
        )

        logger.info("Confusion Matrix:")
        cm = confusion_matrix(labels[test_mask].numpy(), test_preds[test_mask].numpy())
        print("                  Predicted")
        print("                Normal  Fraud")
        print(f"Actual Normal     {cm[0, 0]:4d}   {cm[0, 1]:4d}")
        print(f"       Fraud      {cm[1, 0]:4d}   {cm[1, 1]:4d}")

        accuracy = (test_preds[test_mask] == labels[test_mask]).float().mean()
        logger.info(f"Overall Test Accuracy: {accuracy.item() * 100:.2f}%")
        logger.info("=" * 70)

    # Generate visualization
    visualize_results(labels[test_mask], test_preds[test_mask].numpy())

    return model, features, labels, test_preds


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("BWGNN FRAUD DETECTION DEMO")
    logger.info("=" * 70)

    model, features, labels, predictions = train_and_evaluate()

    logger.info("Demo completed successfully")
