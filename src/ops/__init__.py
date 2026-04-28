"""
Quantized operators — OpQuantConfig-driven, bit-exact equivalent to mx/.

All operators consume OpQuantConfig pipelines (no mx_specs, no MxSpecs).
ObservableMixin provides _emit for analysis (no-op in Phase 3).
"""
from .linear import QuantizedLinear, LinearFunction
from .matmul import MatMulFunction, quantized_matmul
from .bmm import BMMFunction, quantized_bmm
from .conv import (ConvFunction, QuantizedConv1d, QuantizedConv2d,
                   QuantizedConv3d, ConvTransposeFunction,
                   QuantizedConvTranspose1d, QuantizedConvTranspose2d,
                   QuantizedConvTranspose3d)
from .norm import (BatchNormFunction, LayerNormFunction, GroupNormFunction,
                   RMSNormFunction, QuantizedBatchNorm1d, QuantizedBatchNorm2d,
                   QuantizedBatchNorm3d, QuantizedLayerNorm, QuantizedGroupNorm,
                   QuantizedRMSNorm)
from .activations import (SigmoidFunction, TanhFunction, ReLUFunction,
                          ReLU6Function, LeakyReLUFunction, SiLUFunction,
                          GELUFunction, QuantizedSigmoid, QuantizedTanh,
                          QuantizedReLU, QuantizedReLU6, QuantizedLeakyReLU,
                          QuantizedSiLU, QuantizedGELU)
from .softmax import SoftmaxFunction, QuantizedSoftmax
from .pooling import AdaptiveAvgPool2dFunction, QuantizedAdaptiveAvgPool2d
