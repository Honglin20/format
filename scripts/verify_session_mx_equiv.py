"""
Compare Session vs MX accuracy on pre-trained MNIST and Transformer models.

Run: PYTHONPATH=. python scripts/verify_session_mx_equiv.py
"""
import copy, csv, os, sys, urllib.request, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mx
from mx.specs import finalize_mx_specs
from mx import mx_mapping

from src.session import Session, QuantConfig


# ═══════════════════════════════════════════════════════════════════════════
# Configs to test
# ═══════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 32

FORMATS = [
    ("int8",       "mxint8"),
    ("int4",       "mxint4"),
    ("fp8_e4m3",   "mxfp8_e4m3"),
    ("fp8_e5m2",   "mxfp8_e5m2"),
    ("fp4_e2m1",   "mxfp4"),
]

def make_mx_specs(fmt):
    return finalize_mx_specs({
        "w_elem_format": fmt, "a_elem_format": fmt,
        "block_size": BLOCK_SIZE, "bfloat": 16,
        "quantize_backprop": False,
    })

def make_qcfg(fmt):
    return QuantConfig(
        w_format=fmt, a_format=fmt,
        w_granularity="per_block", a_granularity="per_block",
        w_block_size=BLOCK_SIZE, a_block_size=BLOCK_SIZE,
        storage_bits=16, storage_kind="bfloat",
        quantize_nonlinear=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MNIST model & data
# ═══════════════════════════════════════════════════════════════════════════

def load_mnist(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    expected_fp32 = ckpt.get("fp32_test_acc", None)
    return model, expected_fp32

def mnist_test_loader(batch_size=256):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer model & data
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=4, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def load_transformer(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = TransformerClassifier(
        vocab_size=ckpt["vocab_size"], num_classes=ckpt["num_classes"],
        d_model=ckpt["d_model"], nhead=ckpt["nhead"],
        num_layers=ckpt["num_layers"], dim_feedforward=ckpt["dim_feedforward"],
        max_len=ckpt["max_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    expected_fp32 = ckpt.get("fp32_test_acc", None)
    return model, expected_fp32

def agnews_data(vocab_path, max_len, batch_size=128):
    PAD, UNK = "<pad>", "<unk>"
    vocab = torch.load(vocab_path, map_location="cpu", weights_only=False)["vocab"]
    def tok(text):
        ids = [vocab.get(t, vocab[UNK]) for t in text.lower().split()]
        if len(ids) > max_len: ids = ids[:max_len]
        else: ids += [vocab[PAD]] * (max_len - len(ids))
        return ids

    DATA = "/tmp/agnews_data"
    os.makedirs(DATA, exist_ok=True)
    test_path = os.path.join(DATA, "test.csv")
    if not os.path.exists(test_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", test_path)

    texts, labels = [], []
    with open(test_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3: continue
            texts.append(tok(row[1] + " " + row[2]))
            labels.append(int(row[0]) - 1)
            if len(labels) >= 7600: break
    ds = TensorDataset(torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# Eval helpers
# ═══════════════════════════════════════════════════════════════════════════

def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Save/restore torch globals (inject_pyt_ops is irreversible)
# ═══════════════════════════════════════════════════════════════════════════

def _save_torch_state():
    state = {}
    for name in ["matmul", "add", "sub", "mul", "div", "exp", "log", "bmm"]:
        if hasattr(torch, name):
            state["torch." + name] = getattr(torch, name)
    for name in ["Linear", "LayerNorm", "BatchNorm1d", "BatchNorm2d",
                 "ReLU", "GELU", "Softmax", "Sigmoid", "Tanh"]:
        if hasattr(torch.nn, name):
            state["nn." + name] = getattr(torch.nn, name)
    for name in ["linear", "layer_norm", "softmax", "relu", "gelu"]:
        if hasattr(torch.nn.functional, name):
            state["F." + name] = getattr(torch.nn.functional, name)
    return state

def _restore_torch_state(state):
    for key, val in state.items():
        parts = key.split(".")
        if parts[0] == "torch" and len(parts) == 2:
            setattr(torch, parts[1], val)
        elif parts[0] == "nn":
            setattr(torch.nn, parts[1], val)
        elif parts[0] == "F":
            setattr(torch.nn.functional, parts[1], val)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    script_dir = os.path.dirname(__file__)
    wdir = os.path.join(script_dir, "weights")
    failures = []

    for model_name, load_fn, loader_fn in [
        ("MNIST MLP", load_mnist, lambda: mnist_test_loader()),
        ("Transformer AG News", load_transformer, lambda: agnews_data(
            os.path.join(wdir, "transformer_agnews_vocab.pt"), max_len=64)),
    ]:
        print("=" * 72)
        print(f"{model_name}: FP32 → MX vs Session")
        print("=" * 72)

        # 1. Load model & baseline
        wpath = os.path.join(wdir, "mnist_mlp.pt" if "MNIST" in model_name else "transformer_agnews.pt")
        base_model, expected_fp32 = load_fn(wpath)
        loader = loader_fn()

        fp32_acc = accuracy(base_model, loader)
        print(f"\nFP32 accuracy: {fp32_acc:.4f}" +
              (f"  (saved: {expected_fp32:.4f})" if expected_fp32 else ""))

        print(f"\n{'Format':<16} {'FP32':<10} {'MX Acc':<10} {'Sess Acc':<10} {'Match?':<8}")
        print("-" * 58)

        torch_orig = _save_torch_state()  # save before any patching

        for fmt_name, display in FORMATS:
            # ── MX via inject_pyt_ops ──
            _restore_torch_state(torch_orig)
            specs = make_mx_specs(fmt_name)
            mx_mapping.inject_pyt_ops(specs)
            mx_model = copy.deepcopy(base_model).eval()
            mx_acc = accuracy(mx_model, loader)

            # ── Session ──
            _restore_torch_state(torch_orig)
            sess_model = copy.deepcopy(base_model).eval()
            qcfg = make_qcfg(fmt_name)
            session = Session(sess_model, qcfg).quantize()
            session.qmodel.eval()
            sess_acc = accuracy(session.qmodel, loader)

            match = "OK" if abs(mx_acc - sess_acc) < 0.001 else f"Δ={abs(mx_acc-sess_acc):.4f}"
            print(f"{display:<16} {fp32_acc:<10.4f} {mx_acc:<10.4f} {sess_acc:<10.4f} {match:<8}")

            if abs(mx_acc - sess_acc) >= 0.001:
                failures.append(f"{model_name} {display}: MX={mx_acc:.4f} Session={sess_acc:.4f}")

        _restore_torch_state(torch_orig)

    print(f"\n{'=' * 72}")
    if failures:
        print(f"MISMATCHES ({len(failures)}):")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("ALL PASS — Session and MX accuracy match (Δ < 0.001) across all formats.")
        sys.exit(0)


if __name__ == "__main__":
    main()
