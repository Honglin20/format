# 多配置对比 (Study)

`Study` 对多个 `QuantConfig` 并行运行 Session，产出 `StudyReport` 做聚合对比。

## 基础用法

```python
from src.session import Study, QuantConfig

configs = [
    QuantConfig(name="int8", w_format="int8"),
    QuantConfig(name="int4", w_format="int4"),
    QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
                a_block_size=32),
]

report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)
```

`Study` 对每个 config 创建一个独立 Session，model 自动 deepcopy 互不干扰。

## 终端输出

```python
report.print_summary()

# 输出：
# ======================================================================
#   Part: int8
# ======================================================================
#   Config    Avg QSNR     Avg MSE      Acc Delta
#   ----------------------------------------------------------------
#   int8      34.21        0.001200     loss=-0.0300
#   int4      22.50        0.005000     loss=-0.1000
#   ...
```

## 控制输出

```python
# 默认输出（QSNR only）
report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)

# 启用完整分析
report = Study(configs, model=model).run(
    calib_data, eval_fn=eval_fn,
    outputs=["qsnr", "distribution"]  # qsnr + distribution 指纹
)
```

## 导出 DataFrame

`report.to_dataframe()` 返回 tidy DataFrame——每行 `(part, config, format, layer, role)`：

```python
df = report.to_dataframe()

# 每个配置的平均 QSNR
print(df.groupby("config")["qsnr_db"].mean())

# 找出质量差的层
print(df[df["qsnr_db"] < 20][["layer", "role", "qsnr_db"]])
```

## 保存 & 重新加载

```python
report.save("results/")

# 从保存的结果加载
from src.report import StudyReport
report = StudyReport.from_file("results/")
```

产出文件结构：

```
results/
├── results.json
├── tables/
│   └── accuracy.csv
└── figures/
    ├── qsnr_comparison.png
    ├── crest_vs_qsnr_input.png
    └── crest_vs_qsnr_weight.png
```

只生成有数据的文件——没跑 eval_fn 不产生 accuracy.csv，没开 DistributionObserver 不产生 crest 散点图。
