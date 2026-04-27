"""
04 — End-to-End Comparison: Comparator, compare_models, compare_sessions.

Shows all three comparison modes with user-defined eval functions.
Run:  PYTHONPATH=. python examples/04_e2e_comparison.py
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from _model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.session import QuantSession
from src.analysis.e2e import Comparator, compare_models, compare_sessions
from src.mapping.quantize_model import quantize_model


def make_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    return OpQuantConfig(input=(scheme,), weight=(scheme,), output=(scheme,))


def make_eval_loader(n_samples=64, batch_size=8, n_classes=3):
    data = torch.randn(n_samples, 128)
    labels = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(data, labels), batch_size=batch_size)


def top1_accuracy(logits, labels):
    return {"top1": (logits.argmax(-1) == labels).float().mean().item()}


def main():
    print("=" * 55)
    print("End-to-End Comparison Tools")
    print("=" * 55)

    dl = make_eval_loader()

    # ── Mode 1: Comparator (manual) ───────────────────────────────
    print("\n1. Comparator — manual control loop")

    fp32 = ToyMLP()
    fp32.eval()

    qmodel = quantize_model(ToyMLP(), cfg=make_cfg())
    qmodel.load_state_dict(fp32.state_dict(), strict=False)
    qmodel.eval()

    cmp = Comparator()
    with cmp, torch.no_grad():
        for inputs, labels in dl:
            fp32_out = fp32(inputs)
            q_out = qmodel(inputs)
            cmp.record(fp32_out, q_out, labels)

    result = cmp.evaluate(top1_accuracy, directions={"top1": "higher"})
    print(f"   fp32  top1:  {result['fp32']['top1']:.4f}")
    print(f"   quant top1:  {result['quant']['top1']:.4f}")
    print(f"   delta:       {result['delta']['top1']:+.4f}")
    print(f"   samples:     {cmp.num_samples}")

    # ── Mode 2: compare_models (auto) ─────────────────────────────
    print("\n2. compare_models — auto mode")

    result2 = compare_models(fp32, qmodel, dl, eval_fn=top1_accuracy)
    print(f"   fp32  top1:  {result2['fp32']['top1']:.4f}")
    print(f"   quant top1:  {result2['quant']['top1']:.4f}")
    print(f"   delta:       {result2['delta']['top1']:+.4f}")

    # ── Mode 3: compare_sessions (multi-session) ──────────────────
    print("\n3. compare_sessions — multiple quantization configs")

    # Shared fp32 model (same weights)
    base_model = ToyMLP()
    base_model.eval()

    # int8 session
    s_int8 = QuantSession(ToyMLP(), make_cfg())
    s_int8.eval()
    s_int8.qmodel.load_state_dict(base_model.state_dict(), strict=False)

    # fp8 session
    fp8_fmt = FormatBase.from_str("fp8_e4m3")
    fp8_scheme = QuantScheme(format=fp8_fmt, granularity=GranularitySpec.per_tensor())
    fp8_cfg = OpQuantConfig(input=(fp8_scheme,), weight=(fp8_scheme,), output=(fp8_scheme,))
    s_fp8 = QuantSession(ToyMLP(), fp8_cfg)
    s_fp8.eval()
    s_fp8.qmodel.load_state_dict(base_model.state_dict(), strict=False)

    # nf4 session
    nf4_fmt = FormatBase.from_str("nf4")
    nf4_scheme = QuantScheme(nf4_fmt, GranularitySpec.per_channel(axis=0))
    nf4_cfg = OpQuantConfig(weight=(nf4_scheme,))
    s_nf4 = QuantSession(ToyMLP(), nf4_cfg)
    s_nf4.eval()
    s_nf4.qmodel.load_state_dict(base_model.state_dict(), strict=False)

    results = compare_sessions(
        {"int8": s_int8, "fp8": s_fp8, "nf4": s_nf4},
        dl,
        eval_fn=top1_accuracy,
        directions={"top1": "higher"},
    )

    print(f"   fp32 baseline top1: {results['fp32']['top1']:.4f}")
    print(f"   {'Config':<8} {'fp32 top1':>10} {'quant top1':>11} {'delta':>10}")
    print(f"   {'-'*8} {'-'*10} {'-'*11} {'-'*10}")
    for name in ["int8", "fp8", "nf4"]:
        r = results[name]
        print(f"   {name:<8} {r['fp32']['top1']:>10.4f} {r['quant']['top1']:>11.4f} {r['delta']['top1']:>+10.4f}")

    # ── Mode 4: Session convenience method ────────────────────────
    print("\n4. QuantSession.compare() — convenience method")

    session = QuantSession(ToyMLP(), make_cfg())
    session.eval()
    result4 = session.compare(dl, eval_fn=top1_accuracy, directions={"top1": "higher"})
    print(f"   fp32  top1:  {result4['fp32']['top1']:.4f}")
    print(f"   quant top1:  {result4['quant']['top1']:.4f}")
    print(f"   delta:       {result4['delta']['top1']:+.4f}")

    print("\n" + "=" * 55)
    print("All comparison modes complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
