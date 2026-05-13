"""Capture full ADR-010 analysis output for example document."""
from __future__ import annotations

import copy, csv, os, urllib.request, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.session import Session, QuantConfig

# ── Data loading ──

AG_NEWS_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
AG_NEWS_TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
DATA_DIR = "/tmp/agnews_data"

os.makedirs(DATA_DIR, exist_ok=True)
for url, name in [(AG_NEWS_TRAIN_URL, "train.csv"), (AG_NEWS_TEST_URL, "test.csv")]:
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, p)

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

def tokenise(text, vocab, max_len=64):
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in text.lower().split()]
    if len(ids) > max_len: ids = ids[:max_len]
    else: ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids

def load_agnews(path, vocab, max_len=64, limit=None):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3: continue
            texts.append(tokenise(row[1] + " " + row[2], vocab, max_len))
            labels.append(int(row[0]) - 1)
            if limit and len(labels) >= limit: break
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# ── Model ──

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class SlowTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Override forward to always use the slow path.

    PyTorch's TransformerEncoderLayer has a fused fast path that extracts
    raw weight tensors from submodules and passes them to a C++ kernel,
    bypassing QuantizedLinear/QuantizedLayerNorm.forward() entirely.
    This prevents observer hooks (_emit) from firing on internal modules.

    The slow path calls self.linear1(), self.norm1(), etc. directly,
    which routes through the Quantized* forward methods and triggers
    observer data collection.
    """
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=4, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = SlowTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.embedding(x); x = self.pos_encoder(x); x = self.transformer(x); x = x.mean(dim=1)
        return self.classifier(x)

def eval_fn(model, data):
    model.eval()
    if isinstance(data, list):
        with torch.no_grad():
            for batch in data: model(batch)
        return {}
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}

# ── Load model & data ──

torch.manual_seed(42)
script_dir = os.path.dirname(__file__)
ckpt = torch.load(os.path.join(script_dir, "weights", "transformer_agnews.pt"), map_location="cpu")
vocab = ckpt["vocab"]
hparams = {k: ckpt[k] for k in ["vocab_size","num_classes","d_model","nhead","num_layers","dim_feedforward","max_len"]}

model = TransformerClassifier(**hparams)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
test_x, test_y = load_agnews(test_path, vocab, max_len=hparams["max_len"], limit=7600)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=128)

train_x_small, _ = load_agnews(train_path, vocab, max_len=hparams["max_len"], limit=512)
calib_loader = DataLoader(TensorDataset(train_x_small, torch.zeros(len(train_x_small), dtype=torch.long)), batch_size=64)
calib_samples = [x for x, _y in calib_loader][:8]

# ── Config ──

cfg = QuantConfig(
    name="int4-pc", w_format="int4", a_format="int4",
    w_granularity="per_channel", a_granularity="per_channel",
    w_axis=-1, a_axis=-1, quantize_nonlinear=True,
)

# ── Step-by-step session ──

print("=" * 80)
print("SESSION: Step-by-step execution (quantize → calibrate → analyze → evaluate)")
print("=" * 80)

session = Session(model, cfg, keep_fp32=True)
session.quantize(calib_data=calib_samples)
session.calibrate(calib_samples)
session.analyze(calib_samples, outputs=["qsnr", "distribution"])
session.evaluate(test_loader, eval_fn)
result = session.result

print("\n--- result.summary() ---")
print(result.summary())
print("\n--- result.accuracy_table() ---")
print(result.accuracy_table())

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: DIAGNOSE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 1: DIAGNOSE — ErrorProvenance")
print("=" * 80)

prov = result.diagnose

print("\n--- prov.summary() — per-role × layer-type aggregation ---")
print(prov.summary())

print("\n--- prov.per_role_table(max_layers=15) — per-layer input/weight/output QSNR ---")
print(prov.per_role_table(max_layers=15))

print("\n--- prov.top_k(5, role='weight') — worst 5 weight layers ---")
for name, q in prov.top_k(5, role="weight"):
    print(f"  {name:<55} QSNR={q:.1f} dB")

print("\n--- prov.top_k(5, role='auto') — worst per-layer role ---")
for name, q in prov.top_k(5, role="auto"):
    print(f"  {name:<55} QSNR={q:.1f} dB")

print("\n--- prov.top_k(5, role='output') — worst 5 output layers ---")
for name, q in prov.top_k(5, role="output"):
    print(f"  {name:<55} QSNR={q:.1f} dB")

print("\n--- prov.error_source_analysis(role='output') — error propagation ---")
print(prov.error_source_analysis(role="output"))

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: CHARACTERIZE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 2: CHARACTERIZE — DistributionDiagnosis")
print("=" * 80)

diag = result.characterize

worst_w = prov.top_k(3, role="weight")
print("\n--- Per-layer distribution profiles (worst 3 by weight QSNR) ---")
for layer_name, qsnr in worst_w:
    print(f"\n--- diag.profile('{layer_name}', role='weight') ---")
    print(diag.profile(layer_name, role="weight"))

print("\n--- diag.causal_analysis() — full causal matrix ---")
print(diag.causal_analysis())

# Classify a specific layer
print("\n--- diag.classify('transformer.layers.0.linear1', role='weight') ---")
print(diag.classify("transformer.layers.0.linear1", role="weight"))

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: PLAN
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 3: PLAN — InterventionPlanner")
print("=" * 80)

planner = result.plan

print("\n--- planner.top_k_boost(k=3, role='weight', target_bits=8) ---")
plan_w = planner.top_k_boost(k=3, role="weight", target_bits=8)
print(plan_w.explain())

print("\n--- planner.top_k_boost(k=3, role='auto', target_bits=8) ---")
plan_auto = planner.top_k_boost(k=3, role="auto", target_bits=8)
print(plan_auto.explain())

print("\n--- planner.recommend(strategy='conservative') ---")
plan_cons = planner.recommend(strategy="conservative")
print(plan_cons.explain())

print("\n--- planner.recommend(strategy='aggressive') ---")
plan_agg = planner.recommend(strategy="aggressive")
print(plan_agg.explain())

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: INTERVENE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 4: INTERVENE — InterventionAccessor")
print("=" * 80)

intervention = result.intervention

print("\n--- intervention.compare() — apply plan_auto ---")
cmp = intervention.compare(model, calib_samples, plan_auto,
                            eval_data=test_loader, eval_fn=eval_fn)
print(cmp.summary())

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STEP 5: VISUALIZATION — SessionPlotAccessor")
print("=" * 80)

print("\n--- result.plot methods available ---")
plot_methods = [m for m in dir(result.plot) if not m.startswith('_')]
print(f"  {len(plot_methods)} methods: {plot_methods}")

for method in ["per_role_qsnr_bars", "depth_decay", "propagation_dag",
               "error_waterfall", "local_vs_accum_scatter", "layer_histogram",
               "channel_heterogeneity", "crest_vs_qsnr", "outlier_analysis",
               "per_layer_role_histogram", "role_distribution_comparison"]:
    print(f"\n--- result.plot.{method}() ---")
    try:
        fig = getattr(result.plot, method)()
        print(f"  Returns: {type(fig).__name__}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n--- result.tables.per_role_qsnr() ---")
print(result.tables.per_role_qsnr())

print("\n" + "=" * 80)
print("DONE — all analysis output captured")
print("=" * 80)
