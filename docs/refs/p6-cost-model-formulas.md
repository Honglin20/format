# P6 Coarse Cost Model — 原理与公式

> 本文档是 cost model 的**公式权威来源**。实现必须与此文档的公式完全一致。
> 架构设计与集成方案见 `docs/architecture/007-p6-cost-model.md`。

---

## 1. 模型定位

Coarse cost model 在不依赖真实硬件部署的情况下，根据模型图结构和量化配置，估算延迟和显存占用。

**输入**：PyTorch `nn.Module`（已量化或 FP32）+ 可选的 `DeviceSpec`
**输出**：逐层和总计的 `(latency_us, memory_bytes)`
**精度目标**：与真实硬件测量误差 < 50%

---

## 2. 核心公式

### 2.1 延迟（Roofline Model）

经典的 roofline 将运算分为 compute-bound 和 memory-bound 两类，取两者中的瓶颈：

```
latency = max(compute_time, memory_time) × kernel_overhead
```

展开：

```
compute_time  = total_flops / (peak_flops × utilization)
memory_time   = total_bytes / (bandwidth × utilization)
latency_us    = max(compute_time, memory_time) × kernel_overhead × 10^6
```

其中：

| 符号 | 含义 | 单位 | 来源 |
|---|---|---|---|
| `total_flops` | 总浮点运算量（含 math + quantize + transform） | FLOPs | §3 |
| `total_bytes` | 总显存读写量 | bytes | §4 |
| `peak_flops` | GPU 峰值算力 | TFLOPS | `defaults.py` |
| `bandwidth` | GPU 显存带宽 | GB/s | `defaults.py` |
| `utilization` | 实际可用峰值比例（0-1），补偿非理想调度 | — | `defaults.py` |
| `kernel_overhead` | 多 kernel launch 的调度损耗（>1.0） | — | `defaults.py` |

Roofline 的原理：当 `compute_time > memory_time`，算子受算力限制（compute-bound，如大矩阵乘法）；反之受带宽限制（memory-bound，如逐元素操作）。量化本身是 memory-bound 任务，roofline 能够捕捉量化位宽降低带来的带宽节省。

### 2.2 显存占用

逐层显存分两部分：

**权重显存**（静态，生命周期 = 整个推理）：

```
weight_memory = weight_num_elem × effective_bits(weight_scheme) / 8
```

如果权重没有量化（scheme = None），`effective_bits = 32`。

**激活值显存**（动态，逐层复用）：

```
activation_memory = input_elem × effective_bits(input_scheme) / 8
                  + output_elem × effective_bits(output_scheme) / 8
```

**模型总计**：

```
model_memory = Σ weight_memory(layer) + max(activation_memory(layer))
```

激活值取 max 而非 sum：假设 sequential execution，上一层的输出被下一层消费后可复用。

---

## 3. 总 FLOPs 分解

```
total_flops = flops_math + flops_quantize + flops_transform
```

### 3.1 数学运算 FLOPs（flops_math）

各算子类型的标准 FLOPs 公式：

| Op | 公式 | 变量 |
|---|---|---|
| Linear | `2 × B × C_in × C_out` | B=batch, C_in=in_features, C_out=out_features |
| Conv2d | `2 × B × C_out × H_out × W_out × C_in × kH × kW` | 标准卷积 FLOPs 计数 |
| Conv1d | `2 × B × C_out × L_out × C_in × kW` | |
| ConvTranspose2d | `2 × B × C_out × H_out × W_out × C_in × kH × kW` | 对称于 Conv2d |
| BatchNorm | `4 × B × N` | mean+var (2) + affine (2) |
| LayerNorm | `4 × B × N` | 同上 |
| RMSNorm | `2 × B × N` | power (1) + affine (1) |
| GroupNorm | `4 × B × N` | 同 LayerNorm |
| Softmax | `3 × B × N` | max (1) + exp (1) + sum (1) |
| GELU | `8 × B × N` | tanh 近似 |
| SiLU / Sigmoid | `4 × B × N` | exp + div |
| ReLU / ReLU6 / LeakyReLU | `B × N` | 单次比较 |
| Tanh | `5 × B × N` | exp + div + mul |
| AdaptiveAvgPool2d | `B × C × H_in × W_in` | 每个元素一次除法 |
| BMM / Matmul | `2 × B × M × K × N` | batch matmul |
| Elemwise (add / mul) | `B × N` | |

### 3.2 量化 FLOPs（flops_quantize）

量化开销 = 每个量化步骤的 per-element ops + 粒度 reduction 开销。

**Per-element ops 常数**：

| 量化类型 | ops/elem 常数 | 构成 |
|---|---|---|
| 标准 elemwise | `QUANT_OPS_PER_ELEM_BASE = 5` | absmax + log2 + lshift + round + rshift + clamp |
| MX per-block | `QUANT_OPS_PER_ELEM_MX = 8` | shared exp + reshape 开销 + elemwise |
| BF16 截断 | `QUANT_OPS_PER_ELEM_BFLOAT = 2` | mantissa 截断 |
| LookupFormat | `QUANT_OPS_PER_ELEM_LOOKUP = N_levels` | argmin 搜索（NF4 = 16） |

**粒度 reduction 附加 FLOPs**（在 per-element 计算之外追加）：

| 粒度 | 附加 FLOPs | 原理 |
|---|---|---|
| PER_TENSOR | `num_elem` | 全张量 absmax（一次归约） |
| PER_CHANNEL | `num_elem` | 沿 channel_axis absmax |
| PER_BLOCK | `num_elem × 2` | shared_exp 逐 block amax + block reshape 进出 |

**单个量化步骤的 FLOPs**：

```
step_quant_flops = num_elem × per_elem_ops + granularity_additional
```

**算子总 flops_quantize** = 该算子所有非 None scheme 的量化步骤之和。

量化步骤的定义：每调用一次 `quantize(x, scheme)` 算一步。各算子类型的前向步骤数量（从 `src/ops/` 现有实现统计）：

| Op | 量化步骤数 | 说明 |
|---|---|---|
| Linear | 9 | storage(in, w, b, out0, bias_add, out1) + compute(in, w, out) |
| Conv2d | 9 | 同上 |
| BatchNorm | 9 | storage(in, w, b, out) + compute(in, w, b, out) + storage(out) |
| LayerNorm | 10 | storage(in, w, b, out0, normed, out1) + compute(in, w, b, out) + storage(out) |
| RMSNorm | 7 | storage(in, w, out) + compute(in, w, out) + storage(out) |
| GroupNorm | 10 | 同 LayerNorm |
| Softmax | 5 | storage(in, out0, out1) + compute(in, out) |
| GELU / SiLU / ReLU | 5 | storage(in, out0) + compute(in, out) + storage(out) |
| AdaptiveAvgPool2d | 3 | storage(in, out) + compute(in) |
| Elemwise | 5 | storage(x, y, out0) + compute(x, y) + storage(out) |
| Matmul / BMM | 7 | storage(in, w, out0) + compute(in, w, out) + storage(out) |

### 3.3 Transform FLOPs（flops_transform）

Transform 在每个使用它的量化步骤中执行 forward + inverse：

| Transform | ops/elem | 原理 |
|---|---|---|
| Identity | `0` | 无操作 |
| SmoothQuant | `TRANSFORM_OPS_PER_ELEM_DEFAULT = 2` | `x / scale`（fwd）+ `x_q × scale`（inv） |
| PreScale | `TRANSFORM_OPS_PER_ELEM_DEFAULT = 2` | `x × scale`（fwd）+ `x_q / scale`（inv） |
| Hadamard | `TRANSFORM_OPS_PER_ELEM_HADAMARD × log₂(N)` | Walsh-Hadamard 蝶形运算，N = padded dim |

每个步骤：

```
step_transform_flops = num_elem × transform_ops_per_elem
```

算子总 transform FLOPs = 所有使用非 Identity transform 的步骤之和。

---

## 4. 显存读写量公式

### 4.1 有效位宽

量化后的存储位宽按格式折算。Per-block MX 格式额外附加 shared exponent 开销：

```
effective_bits = elem_bits

if granularity == PER_BLOCK:
    effective_bits += MX_SCALE_BITS / block_size
```

| 格式 × 粒度 | elem_bits | block_size | scale_bits | effective_bits |
|---|---|---|---|---|
| FP32 | 32 | — | — | 32.00 |
| BF16 / FP16 | 16 | — | — | 16.00 |
| INT8 per-tensor | 8 | — | — | 8.00 |
| INT8 per-channel | 8 | — | — | 8.00 |
| INT8 MX block=32 | 8 | 32 | 8 | **8.25** |
| FP8 MX block=32 | 8 | 32 | 8 | **8.25** |
| INT4 MX block=32 | 4 | 32 | 8 | **4.25** |
| NF4 per-channel | 4 | — | — | 4.00 |

`MX_SCALE_BITS = 8`（E8M0 格式的共享指数）。

### 4.2 单次量化步骤的读写

```
quant_read  = num_elem × 4                  # 从显存读入 FP32
quant_write = num_elem × effective_bits / 8  # 写回量化值
```

读入始终按 FP32（PyTorch 容器）；写出按量化位宽（建模 packed 存储）。

### 4.3 数学运算的读写

```
math_read  = (input_elem + weight_elem + bias_elem) × 4   # FP32 读入
math_write = output_elem × 4                                # FP32 写出
```

对于 matmul 类算子，input 和 weight 从显存读入，output 写入。

### 4.4 算子总计

```
bytes_read  = math_read  + Σ step_read   （所有非 None scheme 步骤）
bytes_write = math_write + Σ step_write  （所有非 None scheme 步骤）
```

---

## 5. 可配置常数

所有可调参数集中定义在 `src/cost/defaults.py`。默认值按 A100 标定：

```python
# ── Device ─────────────────────────────────────────────────
DEFAULT_PEAK_FLOPS_FP32 = 19.5       # TFLOPS (A100)
DEFAULT_MEMORY_BANDWIDTH_GBS = 2039  # GB/s (A100 80GB)
DEFAULT_DEVICE_MEMORY_GB = 80.0

# ── Roofline correction ────────────────────────────────────
DEFAULT_UTILIZATION = 0.4            # 0.0-1.0，非理想调度的折扣因子
DEFAULT_KERNEL_OVERHEAD = 1.3        # >1.0，多 kernel launch 的调度惩罚

# ── Quantize per-element ops ───────────────────────────────
QUANT_OPS_PER_ELEM_BASE = 5          # absmax + log2 + lshift + round + rshift + clamp
QUANT_OPS_PER_ELEM_MX = 8            # 同上 + shared exp + reshape 额外开销
QUANT_OPS_PER_ELEM_BFLOAT = 2        # mantissa 截断
QUANT_OPS_PER_ELEM_LOOKUP = 16       # NF4: argmin 搜索 16 levels

# ── Transform per-element ops ──────────────────────────────
TRANSFORM_OPS_PER_ELEM_DEFAULT = 2   # 乘除各一次（fwd + inv）
TRANSFORM_OPS_PER_ELEM_HADAMARD = 2  # log₂(N) 蝶形运算系数

# ── MX constants ───────────────────────────────────────────
MX_SCALE_BITS = 8                    # E8M0 shared exponent
```

---

## 6. 参考

- Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM.
- `docs/architecture/007-p6-cost-model.md` — 架构设计、Session 集成、包结构
