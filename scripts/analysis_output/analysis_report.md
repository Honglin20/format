# 4-bit Format Analysis Report — Shakespeare GPT

- **Model**: MiniGPT(d=192, h=3, L=4, T=128)
- **Parameters**: 1,822,464
- **FP32 Validation PPL**: 7.9474
- **Calibration batches**: 4

## Part 1: MXINT Precision Comparison (W8A8 / W4A8 / W4A4)

| Config | PPL | ΔPPL | QSNR (local) | QSNR (accum) |
|--------|-----|------|--------------|---------------|
| FP32 baseline | 7.9474 | — | — | — |
| MXINT-W8A8 | 7.9517 | +0.0044 | 43.29 dB | 39.26 dB |
| MXINT-W4A8 | 8.0195 | +0.0721 | 23.75 dB | 20.21 dB |
| MXINT-W4A4 | 8.4283 | +0.4809 | 19.00 dB | 14.83 dB |

## Part 2: Root Cause Analysis

### Top-10 Worst Layers (local QSNR)

| Layer | QSNR (dB) |
|-------|-----------|
| blocks.0.attn.qkv | 15.31 |
| blocks.0.mlp.0 | 15.61 |
| blocks.1.mlp.0 | 16.29 |
| blocks.2.mlp.0 | 16.80 |
| blocks.1.attn.qkv | 17.72 |
| blocks.2.mlp.2 | 17.93 |
| blocks.1.mlp.2 | 18.17 |
| blocks.3.mlp.0 | 18.35 |
| blocks.3.mlp.2 | 19.21 |
| blocks.2.attn.qkv | 19.27 |

### Top-10 Worst Layers (accumulated)

| Layer | QSNR (dB) |
|-------|-----------|
| blocks.3.attn.proj | 10.99 |
| blocks.3.mlp.0 | 11.83 |
| blocks.2.mlp.2 | 12.64 |
| blocks.3.mlp.2 | 12.83 |
| blocks.2.mlp.0 | 13.44 |
| blocks.2.attn.proj | 13.59 |
| blocks.1.mlp.2 | 13.84 |
| blocks.1.mlp.0 | 14.42 |
| blocks.1.attn.proj | 14.89 |
| blocks.0.mlp.0 | 15.37 |

### Per-Layer Per-Role QSNR Attribution

| Layer | Role | QSNR (dB) | Degradation Type |
|-------|------|-----------|-----------------|
| blocks.3.mlp.2               | input  |   15.02 | no_data            |
| blocks.0.mlp.2               | input  |   15.12 | no_data            |
| blocks.2.mlp.2               | input  |   15.13 | no_data            |
| blocks.0.attn.qkv            | output |   15.31 | no_data            |
| blocks.1.mlp.2               | input  |   15.39 | no_data            |
| blocks.0.mlp.0               | output |   15.61 | no_data            |
| blocks.1.mlp.0               | output |   16.29 | no_data            |
| blocks.2.mlp.0               | output |   16.80 | no_data            |
| blocks.1.attn.qkv            | output |   17.72 | no_data            |
| blocks.0.mlp.0               | input  |   17.79 | no_data            |
| blocks.3.mlp.0               | input  |   17.85 | no_data            |
| blocks.2.mlp.2               | output |   17.93 | no_data            |
| blocks.2.attn.qkv            | input  |   17.94 | no_data            |
| blocks.0.attn.qkv            | input  |   17.95 | no_data            |
| blocks.3.attn.qkv            | input  |   17.96 | no_data            |
| blocks.1.attn.qkv            | input  |   17.96 | no_data            |
| blocks.2.mlp.0               | input  |   17.98 | no_data            |
| head                         | input  |   18.00 | no_data            |
| blocks.1.mlp.0               | input  |   18.00 | no_data            |
| blocks.0.attn.proj           | input  |   18.02 | no_data            |
| ... and 31 more rows |

## Part 3: MXFP / NF4 Cross-Format Comparison

| Format | PPL | ΔPPL | QSNR (local) | QSNR (accum) |
|--------|-----|------|--------------|---------------|
| FP32 baseline | 7.9474 | — | — | — |
| MXINT-4      | 8.4283 | +0.4809 | 19.00 dB | 14.83 dB |
| MXFP-4       | 8.4081 | +0.4607 | 19.27 dB | 14.49 dB |
| NF4-W        | 8.0696 | +0.1222 | 18.71 dB | 20.55 dB |
| NF4-WA       | 11.3729 | +3.4255 | 16.74 dB | 9.09 dB |
| MXFP-8       | 7.9679 | +0.0206 | 31.73 dB | 27.35 dB |

## Part 4: Granularity × Sparse Cross-Sweep

| Granularity  | r=0.0   | r=0.01  | r=0.05  | r=0.1   | r=0.2   |
|--------------|---------|---------|---------|---------|---------|
| tensor       |     nan | 17.9105 |     nan |     nan |     nan |
| channel      | 18.6648 | 14.3009 | 12.9515 | 12.6120 | 12.5367 |
| block        |  8.4283 |  8.2966 |  8.2966 |  8.1547 |  8.1888 |

  tensor        best r=0.01  PPL=17.9105
  channel       best r=0.20  PPL=12.5367
  block         best r=0.10  PPL=8.1547

  per_block(32) r=0.00 baseline PPL:  8.4283
  per_tensor best (r=0.01): PPL=17.9105
  per_channel best (r=0.20): PPL=12.5367
  per_block(32)  best (r=0.10): PPL=8.1547

  Can per_tensor + sparse match per_block(32)? NO
    (per_tensor best 17.9105 vs per_block baseline 8.4283; threshold = 8.8497)

  Optimal sparse degree by granularity:
    tensor        r=0.01  (PPL=17.9105)
    channel       r=0.20  (PPL=12.5367)
    block         r=0.10  (PPL=8.1547)

## Conclusions

[TODO: Fill in after full run]
