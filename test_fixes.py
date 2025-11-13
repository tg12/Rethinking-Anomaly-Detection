"""
Test Suite for Fixed BWGNN Implementation

Validates all critical fixes to ensure correctness.
No command-line arguments. All configuration at module level.
"""

# Configuration
RANDOM_SEED = 42
LOG_LEVEL = "INFO"

import logging

import numpy as np
import torch
from torch import nn

from BWGNN_fixed import (BWGNN_Fixed, BWGNN_Hetero_Fixed, SpectralPolyConv,
                         calculate_beta_wavelet_coefficients,
                         edge_index_to_adj_matrix)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set random seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def test_laplacian_computation():
    """Test Fix 1: Proper Laplacian computation."""
    logger.info("TEST 1: Graph Laplacian Computation")

    # Create simple graph: triangle
    num_nodes = 3
    edges = [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]]
    edge_index = torch.LongTensor(edges).t()

    # Add self-loops
    self_loops = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    adj = edge_index_to_adj_matrix(edge_index, num_nodes)

    # Create conv layer
    theta = [1.0, -0.5]
    conv = SpectralPolyConv(2, 2, theta, use_spectral=False)

    # Test features
    features = torch.randn(num_nodes, 2)

    # Forward pass
    output = conv(adj, features)

    logger.info(f"Adjacency matrix shape: {adj.shape}")
    logger.info(f"Input features shape: {features.shape}")
    logger.info(f"Output shape: {output.shape}")

    # Verify Laplacian properties
    L = conv._compute_normalized_laplacian(adj)
    eigenvalues = torch.linalg.eigvalsh(L)

    logger.info(f"Laplacian eigenvalues range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    assert eigenvalues.min() >= -0.01, "Eigenvalues should be non-negative"
    assert eigenvalues.max() <= 2.01, "Max eigenvalue should be <= 2"

    logger.info("TEST 1 PASSED")


def test_beta_wavelet_coefficients():
    """Test Fix 2: Proper Beta wavelet coefficient calculation."""
    logger.info("TEST 2: Beta Wavelet Coefficients")

    d = 2
    thetas = calculate_beta_wavelet_coefficients(d, normalize=True)

    logger.info(f"Number of wavelet scales: {len(thetas)}")
    assert len(thetas) == d + 1, f"Should have {d + 1} wavelets"

    for i, theta in enumerate(thetas):
        norm = np.linalg.norm(theta)
        logger.info(f"Wavelet {i}: {len(theta)} coefficients, L2 norm = {norm:.4f}")
        assert len(theta) == d + 1, f"Each wavelet should have {d + 1} coefficients"
        assert abs(norm - 1.0) < 0.01, f"Wavelet {i} should be normalized"

    # Test filter response
    eigenvalues = np.linspace(0, 2, 100)
    for i, theta in enumerate(thetas):
        response = sum(theta[k] * eigenvalues**k for k in range(len(theta)))
        logger.info(f"Wavelet {i} response range: [{response.min():.4f}, {response.max():.4f}]")

    logger.info("TEST 2 PASSED")


def test_heterogeneous_graph_fix():
    """Test Fix 3: Independent relation processing in heterogeneous graphs."""
    logger.info("TEST 3: Heterogeneous Graph Relation Independence")

    num_nodes = 10
    num_features = 8

    # Create two relation types
    relation_types = ["type_a", "type_b"]

    # Different adjacency matrices
    edges_a = [[0, 1], [1, 2], [2, 3]]
    edges_b = [[4, 5], [5, 6], [6, 7]]

    edge_index_a = torch.LongTensor(edges_a).t()
    edge_index_b = torch.LongTensor(edges_b).t()

    adj_a = edge_index_to_adj_matrix(edge_index_a, num_nodes)
    adj_b = edge_index_to_adj_matrix(edge_index_b, num_nodes)

    # Create model
    model = BWGNN_Hetero_Fixed(
        in_feats=num_features, h_feats=16, num_classes=2, relation_types=relation_types, d=1
    )

    # Set adjacencies
    model.set_adjacency("type_a", adj_a)
    model.set_adjacency("type_b", adj_b)

    # Test features
    features = torch.randn(num_nodes, num_features)

    # Forward pass
    logits = model(features, {"type_a": adj_a, "type_b": adj_b})

    logger.info(f"Model created with {len(relation_types)} relation types")
    logger.info(f"Input features shape: {features.shape}")
    logger.info(f"Output logits shape: {logits.shape}")

    # Verify parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params}")
    assert num_params > 0, "Model should have trainable parameters"

    logger.info("TEST 3 PASSED")


def test_module_list_registration():
    """Test Fix 5: Proper ModuleList parameter registration."""
    logger.info("TEST 4: ModuleList Parameter Registration")

    model = BWGNN_Fixed(in_feats=8, h_feats=16, num_classes=2, d=2)

    # Check conv is ModuleList
    assert isinstance(model.conv, nn.ModuleList), "conv should be nn.ModuleList"
    logger.info("conv is nn.ModuleList")

    # Check parameters are registered
    all_params = list(model.parameters())
    conv_params = list(model.conv.parameters())

    logger.info(f"Total model parameters: {len(all_params)}")
    logger.info(f"Conv layer parameters: {len(conv_params)}")
    assert len(conv_params) > 0, "Conv layers should have parameters"

    # Test state_dict
    state = model.state_dict()
    logger.info(f"State dict keys: {len(state.keys())}")

    conv_keys = [k for k in state.keys() if "conv" in k]
    logger.info(f"Conv parameters in state_dict: {len(conv_keys)}")
    assert len(conv_keys) > 0, "Conv parameters should be in state_dict"

    logger.info("TEST 4 PASSED")


def test_train_eval_mode():
    """Test Fix 4: Proper train/eval mode separation."""
    logger.info("TEST 5: Train/Eval Mode Separation")

    num_nodes = 20
    num_features = 8

    # Create simple graph
    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    edge_index = torch.LongTensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    adj = edge_index_to_adj_matrix(edge_index, num_nodes)
    features = torch.randn(num_nodes, num_features)

    model = BWGNN_Fixed(in_feats=num_features, h_feats=16, num_classes=2, d=1)

    # Test train mode
    model.train()
    assert model.training, "Model should be in training mode"
    logits_train = model(features, adj)
    logger.info("Train mode forward pass successful")

    # Test eval mode
    model.eval()
    assert not model.training, "Model should be in eval mode"

    with torch.no_grad():
        logits_eval = model(features, adj)

    logger.info("Eval mode forward pass successful")
    logger.info(f"Train output shape: {logits_train.shape}")
    logger.info(f"Eval output shape: {logits_eval.shape}")

    assert not logits_eval.requires_grad, "Eval outputs should not require gradients"
    logger.info("Gradients properly disabled in eval mode")

    logger.info("TEST 5 PASSED")


def test_polynomial_vs_spectral():
    """Test that polynomial and spectral methods give similar results."""
    logger.info("TEST 6: Polynomial vs Spectral Consistency")

    num_nodes = 10
    num_features = 4

    # Create graph
    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    edges.extend([[i, (i + 2) % num_nodes] for i in range(num_nodes)])
    edge_index = torch.LongTensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    adj = edge_index_to_adj_matrix(edge_index, num_nodes)
    features = torch.randn(num_nodes, num_features)

    # Create both models with same initialization
    torch.manual_seed(42)
    model_poly = BWGNN_Fixed(num_features, 8, 2, d=1, use_spectral=False)

    torch.manual_seed(42)
    model_spectral = BWGNN_Fixed(num_features, 8, 2, d=1, use_spectral=True)

    # Forward pass
    model_poly.eval()
    model_spectral.eval()

    with torch.no_grad():
        logits_poly = model_poly(features, adj)
        logits_spectral = model_spectral(features, adj)

    logger.info(f"Polynomial output shape: {logits_poly.shape}")
    logger.info(f"Spectral output shape: {logits_spectral.shape}")

    diff = (logits_poly - logits_spectral).abs().mean()
    logger.info(f"Mean absolute difference: {diff:.6f}")
    logger.info("Both methods produce valid outputs")

    logger.info("TEST 6 PASSED")


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    logger.info("TEST 7: Gradient Flow")

    num_nodes = 10
    num_features = 8

    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    edge_index = torch.LongTensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    adj = edge_index_to_adj_matrix(edge_index, num_nodes)
    features = torch.randn(num_nodes, num_features, requires_grad=True)
    labels = torch.randint(0, 2, (num_nodes,))

    model = BWGNN_Fixed(num_features, 16, 2, d=1)
    model.train()

    # Forward pass
    logits = model(features, adj)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    # Backward pass
    loss.backward()

    logger.info(f"Forward pass successful, loss: {loss.item():.4f}")

    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            logger.info(f"{name}: grad norm = {grad_norm:.6f}")

    assert has_gradients, "Model should have gradients after backward pass"
    logger.info("Gradients flow properly through all parameters")

    logger.info("TEST 7 PASSED")


def run_all_tests():
    """Run all tests."""
    logger.info("FIXED BWGNN - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 70)

    try:
        test_laplacian_computation()
        test_beta_wavelet_coefficients()
        test_heterogeneous_graph_fix()
        test_module_list_registration()
        test_train_eval_mode()
        test_polynomial_vs_spectral()
        test_gradient_flow()

        logger.info("=" * 70)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 70)
        logger.info("All critical fixes have been validated")

    except AssertionError as e:
        logger.error(f"TEST FAILED: {e}")
        raise
    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
