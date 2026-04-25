"""
Quantized operators — OpQuantConfig-driven, bit-exact equivalent to mx/.

All operators consume OpQuantConfig pipelines (no mx_specs, no MxSpecs).
ObservableMixin provides _emit for analysis (no-op in Phase 3).
"""
from .linear import QuantizedLinear, LinearFunction
from .matmul import MatMulFunction, quantized_matmul
from .bmm import BMMFunction, quantized_bmm
