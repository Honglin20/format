"""
07 — Pre-Scale + LSQ Gradient-Based Optimization.

PreScaleTransform applies a learnable scale before quantization and
inverses after.  LSQ (Learned Step Size Quantization) optimizes these
scales via gradient descent to minimize MSE against fp32 layer outputs.
PoT (power-of-two) mode constrains scales to powers of 2 for bit-shift
hardware efficiency.

Run:  PYTHONPATH=. python examples/07_pre_scale.py
"""
import copy

import torch

from _model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.session import QuantSession
from src.transform.pre_scale import PreScaleTransform
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer


def main():
    print("=" * 55)
    print("Pre-Scale + LSQ Optimization")
    print("=" * 55)

    # ═════════════════════════════════════════════════════════════════
    # 1. PreScaleTransform — manual usage
    # ═════════════════════════════════════════════════════════════════
    print("\n1. PreScaleTransform — manual usage")

    # PreScaleTransform holds a *reference* to a tensor (not a copy).
    # This means updates to the tensor automatically take effect.
    scale = torch.tensor([2.0, 0.5, 1.0])
    ps = PreScaleTransform(scale=scale)
    x = torch.ones(3, 4)
    out = ps.forward(x)
    back = ps.inverse(out)

    print(f"   scale = {scale.tolist()}")
    print(f"   forward(ones) = {out[:, 0].tolist()}  (broadcasted per-channel)")
    print(f"   round-trip error: {(x - back).abs().max().item():.2e}")

    # PoT mode: scale is projected to nearest power-of-two before use
    ps_pot = PreScaleTransform(scale=torch.tensor([3.0, 0.3, 7.0]), pot=True)
    out_pot = ps_pot.forward(torch.ones(3, 4))
    print(f"   PoT([3.0, 0.3, 7.0]) → {out_pot[:, 0].tolist()}  "
          f"(projected to 2**round(log2(s)))")
    print(f"   invertible: {ps.invertible}  (inverse always defined)")

    # ═════════════════════════════════════════════════════════════════
    # 2. QuantSession.initialize_pre_scales()
    # ═════════════════════════════════════════════════════════════════
    print("\n2. QuantSession.initialize_pre_scales()")

    fp32_model = ToyMLP()
    fp32_model.eval()

    i8f = FormatBase.from_str("int8")
    i8s = QuantScheme(i8f, GranularitySpec.per_tensor())
    cfg = OpQuantConfig(input=i8s, weight=i8s, output=i8s)

    session = QuantSession(copy.deepcopy(fp32_model), cfg=cfg)
    session.eval()

    calib = [torch.randn(8, 128) for _ in range(4)]
    count = session.initialize_pre_scales(calib, init="ones")
    print(f"   Created {count} _pre_scale buffers (tensor ones)")
    print(f"   Pot=False")
    for n, m in session.qmodel.named_modules():
        if hasattr(m, "_pre_scale"):
            print(f"     {n}._pre_scale = {m._pre_scale.flatten().tolist()}")

    # ═════════════════════════════════════════════════════════════════
    # 3. LSQ Optimization (fp32)
    # ═════════════════════════════════════════════════════════════════
    print("\n3. LSQ Optimization (fp32 pre-scales)")

    def mse(qm):
        with torch.no_grad():
            return (fp32_model(torch.randn(8, 128)) -
                    qm(torch.randn(8, 128))).pow(2).mean().item()

    mse_before = mse(session.qmodel)
    opt = LayerwiseScaleOptimizer(num_steps=50, num_batches=2,
                                  optimizer="adam", lr=1e-3)
    result = session.optimize_scales(opt, calib)
    mse_after = mse(session.qmodel)
    delta = (1 - mse_after / mse_before) * 100 if mse_before > 0 else 0
    print(f"   MSE before LSQ:  {mse_before:.6f}")
    print(f"   MSE after LSQ:   {mse_after:.6f}  ({delta:+.1f}%)")
    for name, scale in sorted(result.items()):
        print(f"     {name}: {[f'{v:.4f}' for v in scale.flatten().tolist()]}")

    # ═════════════════════════════════════════════════════════════════
    # 4. LSQ with PoT constraint
    # ═════════════════════════════════════════════════════════════════
    print("\n4. LSQ with PoT constraint")
    print("   PoT projects to 2**round(log2(s)) after each optimizer step.")
    print("   This guarantees bit-shift multiplication (hardware-friendly).")

    session2 = QuantSession(copy.deepcopy(fp32_model), cfg=cfg)
    session2.eval()
    session2.initialize_pre_scales(calib, init="ones", pot=True)

    mse_before2 = mse(session2.qmodel)
    opt2 = LayerwiseScaleOptimizer(num_steps=50, num_batches=2,
                                   optimizer="adam", lr=1e-3, pot=True)
    result2 = session2.optimize_scales(opt2, calib)
    mse_after2 = mse(session2.qmodel)
    delta2 = (1 - mse_after2 / mse_before2) * 100 if mse_before2 > 0 else 0
    print(f"   MSE before LSQ:  {mse_before2:.6f}")
    print(f"   MSE after LSQ:   {mse_after2:.6f}  ({delta2:+.1f}%)")

    # Verify PoT property
    all_pot = all(
        torch.allclose(s, 2 ** torch.round(torch.log2(s)))
        for s in result2.values()
    )
    print(f"   All PoT: {all_pot}")
    for name, scale in sorted(result2.items()):
        is_pot = torch.allclose(scale, 2 ** torch.round(torch.log2(scale)))
        print(f"     {name}: {[f'{v:.4f}' for v in scale.flatten().tolist()]}  "
              f"PoT={is_pot}")

    # Note on PoT trade-off
    print("\n   Note: PoT constrains the optimization space (only discrete")
    print("   powers of 2 are allowed). This may reduce MSE improvement")
    print("   compared to fp32 pre-scales, but guarantees bit-shift scaling")
    print("   for hardware efficiency.")

    # ═════════════════════════════════════════════════════════════════
    # 5. E2E: initialize → optimize → compare
    # ═════════════════════════════════════════════════════════════════
    print("\n5. E2E comparison (session.compare)")

    from torch.utils.data import DataLoader, TensorDataset

    x_data = torch.randn(64, 128)
    y_data = torch.randint(0, 3, (64,))
    dl = DataLoader(TensorDataset(x_data, y_data), batch_size=8)

    def acc(logits, labels):
        return {"acc": (logits.argmax(-1) == labels).float().mean().item()}

    r = session.compare(dl, eval_fn=acc, directions={"acc": "higher"})
    print(f"   fp32 acc: {r['fp32']['acc']:.4f}")
    print(f"   quant acc: {r['quant']['acc']:.4f}")
    print(f"   delta: {r['delta']['acc']:+.4f}")

    # ═════════════════════════════════════════════════════════════════
    # 6. API summary
    # ═════════════════════════════════════════════════════════════════
    print("\n6. API summary")
    print(f"   {'Method':<30} {'Description'}")
    print(f"   {'-'*30} {'-'*40}")
    for method, desc in [
        ("session.initialize_pre_scales()", "Create _pre_scale buffers + transforms"),
        ("session.optimize_scales(opt, data)", "Layer-wise LSQ gradient optimization"),
        ("LayerwiseScaleOptimizer(pot=True)", "PoT projected gradient descent"),
        ("PreScaleTransform(scale, pot=...)", "Reference-based transform in QuantScheme"),
    ]:
        print(f"   {method:<30} {desc}")

    print("\n" + "=" * 55)
    print("Pre-scale + LSQ example complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
