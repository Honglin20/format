"""Experiment configurations for sparse format research.

All parameters that control experimental sweeps live here.
Experiment scripts import these configs — changing a parameter
here changes the experiment without touching any logic.
"""
import torch
from src.scheme.granularity import GranularityMode


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

N_SEEDS = 5
BASE_SEED = 42
DEFAULT_TENSOR_SHAPE = (256, 256)

# ---------------------------------------------------------------------------
# L1: Sparse vs MXINT QSNR comparison
# ---------------------------------------------------------------------------

L1 = {
    "format": "int4",
    "granularity_modes": [
        GranularityMode.PER_TENSOR,
        GranularityMode.PER_CHANNEL,
        GranularityMode.PER_BLOCK,
        GranularityMode.BANK,
    ],
    "sparse_modes": [
        {"label": "dense", "outlier_ratio": 0.0},
        {"label": "sparse", "outlier_ratio": 0.1},
    ],
    "distributions": ["normal", "lognormal", "powerlaw", "real_weight", "real_activation"],
    "tensor_shapes": [(256, 256), (512, 128), (64, 1024)],
    "n_seeds": N_SEEDS,
    "base_seed": BASE_SEED,
    "scale_storage": "pot",  # "pot" or "fp32" — fp32 preserves small amax differences
    # PER_BLOCK settings
    "mxint_block_size": 32,
    # BANK settings
    "bank_size": 16,
}

# ---------------------------------------------------------------------------
# L2: Sparse ratio sweep
# ---------------------------------------------------------------------------

L2 = {
    "format": "int4",
    "granularity_modes": [
        GranularityMode.PER_TENSOR,
        GranularityMode.PER_CHANNEL,
        GranularityMode.BANK,
    ],
    "outlier_ratios": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    "distributions": ["normal", "lognormal", "powerlaw"],
    "tensor_shape": DEFAULT_TENSOR_SHAPE,
    "n_seeds": N_SEEDS,
    "base_seed": BASE_SEED,
    "bank_size": 16,
    "scale_storage": "pot",
}

# ---------------------------------------------------------------------------
# L3: Bank sweet spot
# ---------------------------------------------------------------------------

L3 = {
    "format": "int4",
    "granularity_mode": GranularityMode.BANK,
    "outlier_ratio": 0.1,
    "bank_sizes": [8, 16, 32, 64, 128, 256],
    "tensor_dims": [64, 128, 256, 512, 1024, 2048],
    "fixed_dim": 256,
    "distributions": ["normal", "lognormal"],
    "n_seeds": N_SEEDS,
    "base_seed": BASE_SEED,
    "scale_storage": "pot",
}


# ---------------------------------------------------------------------------
# Helper: generate tensor by distribution name
# ---------------------------------------------------------------------------

def generate_tensor(shape, distribution, seed=None):
    """Generate a tensor with the given shape and distribution type.

    Args:
        shape: Tuple of dimensions.
        distribution: One of "normal", "lognormal", "powerlaw",
                      "real_weight", "real_activation".
        seed: Optional integer seed for reproducibility.

    Returns:
        torch.Tensor with given shape.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if distribution == "normal":
        return torch.randn(shape)
    elif distribution == "lognormal":
        return torch.randn(shape).exp()
    elif distribution == "powerlaw":
        # x ~ powerlaw(alpha=2.5): sample from uniform, transform
        alpha = 2.5
        u = torch.rand(shape)
        x_min = 0.01
        return x_min * (1 - u) ** (-1 / (alpha - 1))
    elif distribution == "real_weight":
        return _get_real_weight(shape)
    elif distribution == "real_activation":
        return _get_real_activation(shape)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ---------------------------------------------------------------------------
# Real tensor extraction (MNIST MLP)
# ---------------------------------------------------------------------------

_real_weight_cache = None
_real_activation_cache = None


def _build_mnist_mlp():
    """Build the MNIST MLP model (same architecture as mnist_hadamard_study.py)."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def _extract_real_tensors():
    """Extract weights and activations from a trained MNIST MLP.

    Tries to load pre-trained weights from scripts/weights/ first;
    falls back to a freshly initialized model.
    """
    global _real_weight_cache, _real_activation_cache
    if _real_weight_cache is not None:
        return _real_weight_cache, _real_activation_cache

    import os
    model = _build_mnist_mlp()

    # Try loading pre-trained weights
    weight_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "scripts", "weights"
    )
    weight_path = os.path.join(weight_dir, "mnist_mlp.pt")
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    model.eval()

    # Collect weight tensors from Linear layers
    weights = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            weights.append(m.weight.data.detach().clone())

    _real_weight_cache = weights

    # Collect activation tensors with a sample input
    activations = []
    x = torch.randn(32, 1, 28, 28)  # batch of 32 MNIST-like images

    def hook_fn(m, inp, out):
        activations.append(out.detach().clone())

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    _real_activation_cache = activations
    return weights, activations


def _get_real_weight(shape):
    """Extract a real weight tensor matching the requested shape if possible."""
    weights, _ = _extract_real_tensors()
    # Find weight with matching ndim and closest element count
    best = None
    for w in weights:
        if w.ndim == len(shape):
            if best is None or abs(w.numel() - (shape[0] * shape[1])) < abs(
                best.numel() - (shape[0] * shape[1])
            ):
                best = w
    if best is None:
        best = weights[0]
    # Reshape or slice to match requested shape
    if best.shape == shape:
        return best.clone()
    # Try reshaping
    if best.numel() >= shape[0] * shape[1]:
        return best.flatten()[: shape[0] * shape[1]].reshape(shape).clone()
    # Pad with zeros
    result = torch.zeros(shape)
    flat_best = best.flatten()
    result.flatten()[: flat_best.numel()] = flat_best
    return result


def _get_real_activation(shape):
    """Extract a real activation tensor matching the requested shape if possible."""
    _, activations = _extract_real_tensors()
    best = None
    for a in activations:
        if a.ndim >= 2:
            # Take first sample from batch
            a_sample = a[0]
            if a_sample.ndim == len(shape):
                if best is None or abs(a_sample.numel() - (shape[0] * shape[1])) < abs(
                    best.numel() - (shape[0] * shape[1])
                ):
                    best = a_sample
    if best is None:
        best = activations[0][0]
    if best.shape == shape:
        return best.clone()
    if best.numel() >= shape[0] * shape[1]:
        return best.flatten()[: shape[0] * shape[1]].reshape(shape).clone()
    result = torch.zeros(shape)
    flat_best = best.flatten()
    result.flatten()[: flat_best.numel()] = flat_best
    return result
