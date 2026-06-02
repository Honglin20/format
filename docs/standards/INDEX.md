# 开发规范 — 怎么做对

> 这些是**具体模块**的接口契约和实现规范。新增任何模块前必须对照。

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [api-design.md](api-design.md) | API 设计约束（哈希基类、post_init、签名稳定性等） | 新增任何公共 API 前 |
| [adding-format.md](adding-format.md) | 新增 Format：接口、注册、测试清单 | 新增量化格式时 |
| [adding-observer.md](adding-observer.md) | 新增 Observer / emit_fn 接入规范 | 新增分析能力时 |
| [adding-transform.md](adding-transform.md) | 新增 Transform：Pre/Post 变换、量化流程集成 | 新增变换时 |
| [onnx-export.md](onnx-export.md) | ONNX export 接入规范（symbolic、Format.export_onnx） | 新增需导出 ONNX 的模块时 |
| [quantization-testing.md](quantization-testing.md) | 量化测试用例编写规范：形状覆盖、block专项、推导前置 | 写任何量化测试前 |
| [e2e-testing.md](e2e-testing.md) | **E2E 测试规范**：三层回归门、准入标准、合并/排除规则、开发流程强制要求 | 提交代码前必读 |
| [role-aware-visualization.md](role-aware-visualization.md) | 可视化与表格的 Role 区分规范：输入/权重/输出显式标注 | 新增/修改绘图或表格前 |
