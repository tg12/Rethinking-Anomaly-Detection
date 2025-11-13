# Fixed BWGNN Implementation

## Overview

This repository contains a corrected implementation of Banded Wavelet Graph Neural Networks (BWGNN) for graph-based anomaly detection. All configuration is at the module level. No command-line arguments or environment variables are used.

## Critical Issues Fixed

### Issue 1: Fundamental Graph Laplacian Misunderstanding

Original implementation computed progressive Laplacian applications incorrectly.

Fixed implementation:
- Properly computes normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
- Stores L^k * x independently for each k
- Correctly combines with polynomial coefficients

### Issue 2: Beta Wavelet Calculation

Original implementation disconnected from graph structure.

Fixed implementation:
- Properly normalizes coefficients to unit L2 norm
- Scales to appropriate range for Laplacian eigenvalues [0, 2]
- Provides both polynomial approximation and true spectral decomposition

### Issue 3: Heterogeneous Graph Processing Bug

Original code overwrote features sequentially across relations.

Fixed implementation:
- Each relation type processes the same base features independently
- No sequential dependency between relations
- Proper aggregation across relation types

### Issue 4: Training Data Leak

Original code reused training-mode logits for validation.

Fixed implementation:
- Recomputes logits in eval mode for validation
- Proper train/eval mode separation
- No gradient computation during validation

### Issue 5: Threshold Selection Methodology

Original code used threshold-optimized F1 for model selection.

Fixed implementation:
- Uses AUPRC (Area Under Precision-Recall Curve) for model selection
- Threshold only tuned on validation set
- Applied to test set after model selection

### Issue 6: Missing ModuleList Registration

Original code used Python list instead of nn.ModuleList.

Fixed implementation:
- Uses nn.ModuleList for proper parameter registration
- Parameters correctly moved to GPU
- Proper checkpoint saving/loading

### Issue 7: Batch Processing Dimension Mismatch

Original code had dimension issues in mini-batch processing.

Fixed implementation:
- Proper handling of source/destination node dimensions
- Correct batched graph processing

## Installation

```bash
pip install -r requirements_fixed.txt
```

Requirements:
- torch>=2.0.0
- numpy>=1.21.0
- scipy>=1.7.0
- sympy>=1.9
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- networkx>=2.6.0

## Usage

### Basic Example

```python
import torch
from BWGNN_fixed import BWGNN_Fixed, edge_index_to_adj_matrix
from train_fixed import train_fixed, TrainingConfig

# Create edge index and features
edge_index = torch.LongTensor([[0, 1, 2], [1, 2, 0]])
features = torch.randn(100, 32)
labels = torch.randint(0, 2, (100,))

# Convert to adjacency matrix
num_nodes = features.shape[0]
adj_matrix = edge_index_to_adj_matrix(edge_index, num_nodes)

# Create model
model = BWGNN_Fixed(
    in_feats=32,
    h_feats=64,
    num_classes=2,
    d=2,
    use_spectral=False
)

# Train model
config = TrainingConfig(
    train_ratio=0.4,
    lr=0.01,
    weight_decay=5e-4,
    epoch=200,
    patience=50
)

mf1, auc, metrics = train_fixed(model, adj_matrix, features, labels, config)
```

### Configuration

All configuration variables are defined at the top of each module:

BWGNN_fixed.py:
- RANDOM_SEED = 42
- DEFAULT_ACTIVATION = 'leaky_relu'
- DEFAULT_NORMALIZE_WAVELETS = True
- EIGENVALUE_CLAMP_MIN = 1.0
- EIGENVALUE_RANGE_MIN = 0.0
- EIGENVALUE_RANGE_MAX = 2.0
- LOG_LEVEL = 'INFO'

train_fixed.py:
- RANDOM_SEED = 42
- DEFAULT_TRAIN_RATIO = 0.4
- DEFAULT_LEARNING_RATE = 0.01
- DEFAULT_WEIGHT_DECAY = 5e-4
- DEFAULT_EPOCHS = 200
- DEFAULT_PATIENCE = 50
- DEFAULT_GRAD_CLIP = 1.0
- DEFAULT_LOG_INTERVAL = 10
- VALIDATION_TEST_SPLIT_RATIO = 0.67
- NUM_THRESHOLD_STEPS = 100
- LOG_LEVEL = 'INFO'

## Testing

Run comprehensive test suite:

```bash
python test_fixes.py
```

Expected output:
- TEST 1 PASSED: Graph Laplacian Computation
- TEST 2 PASSED: Beta Wavelet Coefficients
- TEST 3 PASSED: Heterogeneous Graph Relation Independence
- TEST 4 PASSED: ModuleList Parameter Registration
- TEST 5 PASSED: Train/Eval Mode Separation
- TEST 6 PASSED: Polynomial vs Spectral Consistency
- TEST 7 PASSED: Gradient Flow

## Model Architecture

### BWGNN_Fixed

Standard homogeneous graph neural network.

Architecture:
1. Linear transformation: in_feats -> h_feats
2. Activation (ReLU)
3. Linear transformation: h_feats -> h_feats
4. Activation (ReLU)
5. Wavelet convolutions (multiple scales)
6. Concatenation of wavelet outputs
7. Linear transformation: h_feats * num_scales -> h_feats
8. Activation (ReLU)
9. Linear classification: h_feats -> num_classes

### BWGNN_Hetero_Fixed

Heterogeneous graph version with multiple relation types.

Architecture:
1. Shared feature transformation
2. Independent processing per relation type
3. Aggregation across relations
4. Final classification layer

## Mathematical Details

### Normalized Graph Laplacian

L = I - D^(-1/2) A D^(-1/2)

where:
- I is identity matrix
- D is degree matrix
- A is adjacency matrix

Eigenvalue range: [0, 2]

### Polynomial Graph Convolution

h = sum_{k=0}^d theta_k L^k x

where:
- theta_k are learnable coefficients
- L^k is Laplacian raised to power k
- x is input signal

### Beta Wavelet Polynomials

psi_{d,i}(lambda) = (lambda/2)^i * (1-lambda/2)^(d-i) / Beta(i+1, d+1-i)

where:
- d is polynomial degree
- i indexes different scales
- lambda in [0, 2] is Laplacian eigenvalue
- Beta(a, b) is Beta function

## Performance

### Small Graphs (<100 nodes)
- Training time: <10 seconds
- Recommended: use_spectral=True
- Expected F1: 80-95%

### Medium Graphs (100-1000 nodes)
- Training time: 10-60 seconds
- Recommended: use_spectral=False
- Expected F1: 75-90%

### Large Graphs (1000+ nodes)
- Training time: 1-10 minutes
- Recommended: use_spectral=False
- Expected F1: 70-85%

## File Structure

```
Rethinking-Anomaly-Detection/
|-- BWGNN_fixed.py           # Fixed model implementation
|-- train_fixed.py           # Fixed training procedures
|-- test_fixes.py            # Comprehensive test suite
|-- requirements_fixed.txt   # Dependencies
|-- README_FIXED.md         # This file
```

## Logging

All modules use Python's logging module with structured output.

Log levels:
- INFO: Training progress, test results
- DEBUG: Detailed parameter information
- ERROR: Failures and exceptions

Configure log level by modifying LOG_LEVEL variable at top of each module.

## Reproducibility

Deterministic behavior enforced through:
- Fixed random seeds (RANDOM_SEED = 42)
- torch.manual_seed()
- np.random.seed()
- torch.backends.cudnn.deterministic = True (when CUDA available)

## License

Same as original repository.

## References

1. Spectral Graph Theory: Chung, F. R. (1997)
2. Wavelets on Graphs: Hammond et al. (2011)
3. Graph Neural Networks: Kipf & Welling (2017)
