"""L4: Real model mask generalization (skeleton only).

Measures the QSNR gap between calibration-set and test-set static sparse masks.

This is an interface placeholder — implement when a target model and
calibration data pipeline are available.

Design:
    1. Extract weight / activation tensors from a real model layer.
    2. Generate static sparse masks from calibration samples (S=1..32).
    3. Measure QSNR on hold-out test samples.
    4. Report QSNR_gap = QSNR_calib - QSNR_test.
"""
from typing import List


def run_l4_real_model(
    model_path: str,
    layer_name: str,
    calib_samples: List[int] = None,
    test_samples: int = 100,
    outlier_ratio: float = 0.1,
    granularity: str = "bank",
    bank_size: int = 16,
    output_dir: str = "research/sparse/results",
) -> dict:
    """Measure mask generalization gap on real model tensors.

    This is a skeleton — implement when you have a target model
    and calibration pipeline ready.

    Args:
        model_path: Path to the saved model or model loading function.
        layer_name: Name of the target layer to extract tensors from.
        calib_samples: List of calibration sample counts to sweep.
        test_samples: Number of hold-out test samples.
        outlier_ratio: Fraction of elements marked as outliers.
        granularity: Granularity mode string.
        bank_size: Bank size (only used for BANK granularity).
        output_dir: Directory to write results JSON.

    Returns:
        Dict with keys: calib_qsnr, test_qsnr, gap, mask_stability.
    """
    if calib_samples is None:
        calib_samples = [1, 2, 4, 8, 16, 32]

    raise NotImplementedError(
        "L4 real model generalization is not yet implemented. "
        "Implement when you have a target model and calibration data ready."
    )
