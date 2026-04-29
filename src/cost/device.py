"""DeviceSpec: GPU capability profile for roofline modeling."""
from dataclasses import dataclass
from . import defaults


@dataclass
class DeviceSpec:
    peak_flops_fp32: float    # TFLOPS
    memory_bandwidth_gbs: float  # GB/s
    device_memory_gb: float
    utilization: float = 0.4
    kernel_overhead: float = 1.3

    @staticmethod
    def a100() -> "DeviceSpec":
        return DeviceSpec(
            peak_flops_fp32=defaults.DEFAULT_PEAK_FLOPS_FP32,
            memory_bandwidth_gbs=defaults.DEFAULT_MEMORY_BANDWIDTH_GBS,
            device_memory_gb=defaults.DEFAULT_DEVICE_MEMORY_GB,
            utilization=defaults.DEFAULT_UTILIZATION,
            kernel_overhead=defaults.DEFAULT_KERNEL_OVERHEAD,
        )
