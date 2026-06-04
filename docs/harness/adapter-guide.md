# Adapter 合约

bitx 分析脚本通过 adapter 文件与目标项目解耦。adapter 必须定义三个函数，
分析脚本通过 `importlib` 动态加载。

## 三函数合约

```python
def get_model() -> nn.Module:
    """返回加载好权重的 FP32 模型（eval mode）。"""

def get_eval_fn() -> Callable[[nn.Module, Any], Dict[str, float]]:
    """返回 eval 函数。

    eval_fn(model, data):
      - data is list  → 校准阶段，只跑 forward，return {}
      - data is DataLoader → 评估阶段，return {"accuracy": 0.97}
    """

def get_data() -> Tuple[List[Tensor], Iterable]:
    """返回 (校准数据, 评估数据)。

    calib_data: List[Tensor]，5-8 个 batch
    eval_data: DataLoader 或任何 eval_fn 能处理的 iterable
    """
```

## 为什么是这三个

第一性原理：量化分析只需要三样东西——模型、评估方式、数据。
adapter 把这三样从目标项目的具体实现中抽象出来。

## 生成 adapter 的 agent 策略

1. 读目标项目代码，找到 nn.Module 子类、数据加载、权重文件
2. 检查是否已有 _adapter.py（如果有，验证合约一致性）
3. 生成完整可运行的 adapter，包含所有 import
4. 处理边界情况：权重不存在用 random init，数据不存在用 torchvision 自动下载

## 注意事项

- adapter 里用 `sys.path.insert(0, project_dir)` 确保能 import 目标项目的模块
- 权重加载用 `torch.load(path, map_location="cpu", weights_only=False)`
- `get_data()` 的 calib 数据建议用 eval 集的前 5 个 batch，避免训练集泄漏
- adapter 路径通过 CLI `--adapter` 传入，不需要安装到任何特定位置
