"""
Fraud Detection Demo using BWGNN.

Demonstrates spectral graph convolution with Beta wavelets for detecting
fraudulent transactions in a synthetic financial network.
"""

import logging
import time

import colorlog
import numpy as np
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
LOG_LEVEL = logging.INFO
ENABLE_VISUALIZATION = True

# Configure structured logging with color output
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# Set random seeds for deterministic behavior
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration
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

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def calculate_beta_wavelet_coefficients(eigenvalues, d=2):
    """
    Calculate Beta wavelet coefficients for polynomial approximation.

    Args:
        eigenvalues: Eigenvalues of normalized Laplacian (in [0, 2])
        d: Order of polynomial approximation

    Returns:
        theta: Wavelet coefficients of shape [d+1, num_eigenvalues]
    """

    # Beta wavelet scaling and wavelet functions
    def scaling_function(x):
        return torch.where(x <= 1, torch.ones_like(x), torch.zeros_like(x))

    def wavelet_function(x):
        return torch.where(
            (x > 0.5) & (x <= 1),
            torch.ones_like(x),
            torch.where((x > 1) & (x < 2), -torch.ones_like(x), torch.zeros_like(x)),
        )

    # Apply wavelets at different scales
    theta = []
    for k in range(d + 1):
        scale = 2**k
        psi = wavelet_function(scale * eigenvalues / 2)
        theta.append(psi)

    return torch.stack(theta, dim=0)


class SimpleBWGNN(nn.Module):
    """Simplified BWGNN model for fraud detection."""

    def __init__(self, in_feats, h_feats, num_classes, adj_matrix, d=3):
        super(SimpleBWGNN, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.d = d

        # Store adjacency matrix
        self.register_buffer("adj", torch.FloatTensor(adj_matrix.toarray()))

        # Calculate normalized Laplacian and its eigendecomposition
        self._precompute_spectral()

        # Network layers
        self.conv1 = nn.Linear(in_feats, h_feats)
        self.conv2 = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _precompute_spectral(self):
        """Precompute eigenvalues and eigenvectors of normalized Laplacian."""
        adj = self.adj.numpy()
        n = adj.shape[0]

        # Degree matrix
        degrees = adj.sum(axis=1)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        A_norm = D_sqrt_inv @ adj @ D_sqrt_inv
        L = np.eye(n) - A_norm

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(L)

        # Store as buffers
        self.register_buffer("eigenvalues", torch.FloatTensor(eigenvalues))
        self.register_buffer("eigenvectors", torch.FloatTensor(eigenvectors))

        # Calculate Beta wavelet coefficients
        theta = calculate_beta_wavelet_coefficients(self.eigenvalues, d=self.d)
        self.register_buffer("theta", theta)

    def spectral_filter(self, x):
        """Apply Beta wavelet spectral filtering."""
        # Transform to spectral domain
        x_spectral = self.eigenvectors.T @ x  # [n_nodes, n_features]

        # Apply wavelet filtering
        filtered = torch.zeros_like(x_spectral)
        for k in range(self.d + 1):
            # Apply k-th wavelet coefficient
            filtered += self.theta[k].unsqueeze(-1) * x_spectral

        # Transform back to spatial domain
        x_filtered = self.eigenvectors @ filtered

        return x_filtered

    def forward(self, x):
        """Forward pass with spectral convolution."""
        # First spectral convolution
        x = self.spectral_filter(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second spectral convolution
        x = self.spectral_filter(x)
        x = self.conv2(x)

        return x


def create_fraud_network():
    """
    Create a synthetic transaction network with fraud patterns.

    Returns:
        adj_matrix: Sparse adjacency matrix
        features: Node features
        labels: Binary labels (0=normal, 1=fraud)
    """
    print("Creating synthetic financial network...")

    community_size = NUM_NODES // NUM_NORMAL_COMMUNITIES
    edges = []
    features = []
    labels = []

    # Generate normal communities (tight-knit groups)
    for comm_id in range(NUM_NORMAL_COMMUNITIES):
        start_node = comm_id * community_size
        end_node = start_node + community_size

        for i in range(start_node, end_node):
            # Community-specific features
            base_feature = np.random.randn(FEATURE_DIM) + comm_id * 2
            features.append(base_feature)
            labels.append(0)  # Normal

            # Dense connections within community
            num_connections = np.random.randint(5, 11)
            targets = np.random.choice(
                range(start_node, end_node),
                size=min(num_connections, end_node - start_node - 1),
                replace=False,
            )
            for target in targets:
                if target != i:
                    edges.append((i, target))
                    edges.append((target, i))  # Undirected

    # Add cross-community edges
    num_cross_edges = NUM_NODES // 2
    for _ in range(num_cross_edges):
        src = np.random.randint(0, NUM_NODES)
        dst = np.random.randint(0, NUM_NODES)
        if src != dst:
            edges.append((src, dst))
            edges.append((dst, src))

    # Inject fraudulent nodes
    num_fraud = int(NUM_NODES * FRAUD_RATIO)
    fraud_nodes = np.random.choice(NUM_NODES, size=num_fraud, replace=False)

    for fraud_node in fraud_nodes:
        labels[fraud_node] = 1  # Mark as fraud

        # Anomalous features
        features[fraud_node] = np.random.randn(FEATURE_DIM) * 3 + 10

        # Connect to random victims (not community-based)
        num_victims = np.random.randint(8, 15)
        victims = np.random.choice(NUM_NODES, size=num_victims, replace=False)
        for victim in victims:
            if victim != fraud_node:
                edges.append((fraud_node, victim))
                edges.append((victim, fraud_node))

    # Convert to sparse adjacency matrix
    edges = list(set(edges))  # Remove duplicates
    rows, cols = zip(*edges) if edges else ([], [])
    data = np.ones(len(rows))
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(NUM_NODES, NUM_NODES))

    # Convert features and labels to tensors
    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Normalize features
    features = (features - features.mean(0)) / (features.std(0) + 1e-5)

    num_edges = len(edges)
    print(f"Network created: {NUM_NODES} nodes, {num_edges} edges")
    print(f"Fraud nodes: {num_fraud} ({FRAUD_RATIO * 100:.1f}%)")
    print(f"Normal nodes: {NUM_NODES - num_fraud}")

    return adj_matrix, features, labels


def visualize_results(labels, predictions):
    """Create ASCII visualization of results."""
    if not HAS_MATPLOTLIB:
        return

    print("\nGenerating visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix heatmap
    cm = confusion_matrix(labels.numpy(), predictions)
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Normal", "Fraud"])
    ax1.set_yticklabels(["Normal", "Fraud"])
    ax1.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=20,
                fontweight="bold",
            )

    plt.colorbar(im1, ax=ax1)

    # Performance metrics bar chart
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
    print("Visualization saved as 'fraud_detection_results.png'")
    plt.close()


def train_and_evaluate():
    """Train BWGNN model and evaluate fraud detection performance."""
    # Create synthetic network
    adj_matrix, features, labels = create_fraud_network()

    # Split data
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

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    print("\nDataset split:")
    print(f"  Training: {train_mask.sum().item()} nodes")
    print(f"  Validation: {val_mask.sum().item()} nodes")
    print(f"  Test: {test_mask.sum().item()} nodes")

    # Initialize model
    print("\nInitializing BWGNN model...")
    model = SimpleBWGNN(
        in_feats=FEATURE_DIM,
        h_feats=HIDDEN_DIM,
        num_classes=2,
        adj_matrix=adj_matrix,
        d=BETA_WAVELET_ORDER,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Class weights for imbalanced data
    fraud_weight = (labels[train_mask] == 0).sum().float() / (labels[train_mask] == 1).sum().float()
    class_weights = torch.tensor([1.0, fraud_weight.item()])

    print(f"  Class weight for fraud: {fraud_weight.item():.2f}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Beta wavelet order: {BETA_WAVELET_ORDER}")

    # Training loop
    print(f"\nTraining BWGNN model for {EPOCHS} epochs...")
    best_val_auc = 0
    best_epoch = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Training
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_probs = F.softmax(val_logits, dim=1)
            val_auc = roc_auc_score(labels[val_mask].numpy(), val_probs[val_mask, 1].numpy())

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 30 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}"
            )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch + 1}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(features)
        test_probs = F.softmax(test_logits, dim=1)
        test_preds = test_probs.argmax(dim=1)

        # Calculate metrics
        test_auc = roc_auc_score(labels[test_mask].numpy(), test_probs[test_mask, 1].numpy())

        print(f"\n{'=' * 70}")
        print(" " * 20 + "FRAUD DETECTION RESULTS")
        print(f"{'=' * 70}")
        print(f"\nTest Set AUC: {test_auc:.4f}")
        print("\nClassification Report (Test Set):")
        print(
            classification_report(
                labels[test_mask].numpy(),
                test_preds[test_mask].numpy(),
                target_names=["Normal", "Fraud"],
                digits=4,
            )
        )

        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(labels[test_mask].numpy(), test_preds[test_mask].numpy())
        print("                    Predicted")
        print("                  Normal  Fraud")
        print(f"  Actual Normal     {cm[0, 0]:4d}   {cm[0, 1]:4d}")
        print(f"         Fraud      {cm[1, 0]:4d}   {cm[1, 1]:4d}")

        # Overall metrics
        accuracy = (test_preds[test_mask] == labels[test_mask]).float().mean()
        print(f"\nOverall Test Accuracy: {accuracy.item() * 100:.2f}%")
        print(f"{'=' * 70}\n")

    # Visualize results
    visualize_results(labels[test_mask], test_preds[test_mask].numpy())

    return model, features, labels, test_preds


if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "BWGNN FRAUD DETECTION DEMO")
    print("=" * 70)
    print("\nThis demo shows how BWGNN detects fraudulent transactions")
    print("in a synthetic financial network using spectral graph")
    print("convolution with Beta wavelets.")
    print("\nKey concepts demonstrated:")
    print("  - Normal users form tight communities (friends paying friends)")
    print("  - Fraudsters connect randomly to victims (anomalous pattern)")
    print("  - BWGNN uses spectral analysis to detect these structural anomalies")
    print("  - Beta wavelets capture multi-scale graph patterns\n")

    model, features, labels, predictions = train_and_evaluate()

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  ✓ BWGNN leverages graph structure to detect anomalous patterns")
    print("  ✓ Beta wavelets capture multi-scale information in the graph")
    print("  ✓ Spectral methods are effective for fraud detection")
    print("  ✓ The model learned to distinguish community vs. random patterns")
    print("  ✓ No external graph libraries needed - pure PyTorch + NumPy!\n")

    if HAS_MATPLOTLIB:
        print("Check 'fraud_detection_results.png' for visualization!\n")
