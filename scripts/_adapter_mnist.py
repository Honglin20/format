"""MNIST MLP adapter — test adapter for mxint_error_analysis.py."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_SCRIPT_DIR, "weights", "mnist_mlp.pt")
_DATA_ROOT = "/tmp/mnist_data"


def _build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def get_model() -> nn.Module:
    model = _build_model()
    if os.path.exists(_WEIGHTS_PATH):
        ckpt = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[adapter] No weights found, using random init")
    return model.eval()


def get_eval_fn():
    def eval_fn(model, data):
        model.eval()
        if isinstance(data, list):
            with torch.no_grad():
                for batch in data:
                    model(batch)
            return {}
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in data:
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return {"accuracy": correct / total if total > 0 else 0.0}
    return eval_fn


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(_DATA_ROOT, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    calib = []
    for x, _y in test_loader:
        calib.append(x)
        if len(calib) >= 5:
            break
    return calib, test_loader
