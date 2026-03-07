# PyImgAno 能力与差距评估（面向开发者）

> Last updated: **2026-03-07**（对应仓库版本：`v0.6.37`）

这份文档回答两个问题：

1) **开发者第一次看到项目，最关心的点是否都具备？怎么用？**
2) **和顶尖同类包（PyOD / anomalib 等）相比，差距在哪里？下一步补什么最值？**

---

## 1) 开发者最关心的点：Checklist（以及对应入口）

### ✅ 安装是否“轻量” & 可选依赖是否拆分干净？

- 核心安装：`pip install pyimgano`
  - 目标：CPU baseline + 基础 CLI 可用
- 可选 extras（按需安装）：
  - 深度/torch：`pip install "pyimgano[torch]"`
  - ONNX：`pip install "pyimgano[onnx]"`
  - OpenVINO：`pip install "pyimgano[openvino]"`
  - scikit-image（SSIM/LBP/HOG/Gabor/phase-corr 等）：`pip install "pyimgano[skimage]"`
  - 一揽子：`pip install "pyimgano[all]"`

对应文档：`docs/OPTIONAL_DEPENDENCIES.md`

### ✅ import 边界是否严格（没装 extras 不会一 import 就炸）？

是。核心策略：

- 轻量入口（例如 `import pyimgano` / discovery CLI）不应隐式 import `torch`/`onnxruntime` 等
- 需要 extras 的功能点，在**真正使用到**时才触发 `ImportError`，并给出安装提示（例如 `pip install 'pyimgano[torch]'`）

相关实现：`pyimgano/utils/optional_deps.py`  
（`require(...)` 负责把“缺依赖”变成“可行动的提示”）

### ✅ 离线安全（默认不下载权重）？

是。工业落地里“默认离线/可审计”比“自动下载权重”更重要：

- 多数 CLI 默认 `--no-pretrained`
- 用户需要时再显式 `--pretrained`

对应文档：`docs/WEIGHTS.md`、`docs/CLI_REFERENCE.md`

### ✅ “怎么选算法”：有 baseline 套件 + 小网格搜索吗？

有。推荐从 suite + sweep 入手，避免拍脑袋：

- 查看 suites：`pyimgano-benchmark --list-suites`
- 推荐 suite：`industrial-v4`（core + optional）
- 查看 sweeps：`pyimgano-benchmark --list-sweeps`
- 小网格扫参：`--suite-sweep industrial-feature-small`（新增）

对应文档：`README.md`、`docs/CLI_REFERENCE.md`  
实现位置：`pyimgano/baselines/suites.py`、`pyimgano/baselines/sweeps.py`

### ✅ 工业输出是否“可落地”（JSONL / masks / 缺陷区域）？

是。项目把“部署时真正需要的输出形态”当一等公民：

- `pyimgano-infer` 输出 JSONL
- `--defects` 产出 defect masks + connected components 区域统计（可直接对接 MES/质检系统）

对应文档：`docs/INDUSTRIAL_INFERENCE.md`、`docs/CLI_REFERENCE.md`

### ✅ 可复现 / 可审计（run artifacts / 环境信息）？

是。suite/workbench 会写入 run artifacts（例如 `report.json`、`environment.json`、`per_image.jsonl`），方便追溯与复盘。

对应文档：`docs/WORKBENCH.md`、`docs/EVALUATION_AND_BENCHMARK.md`

### ✅ 可扩展（企业内部模型/算法怎么接入）？

支持插件式扩展（entry points），并提供明确的文档入口。

对应文档：`docs/PLUGINS.md`

---

## 2) PyOD / anomalib 是否“复合包”？anomalib 能不能拆？

### PyOD：更像“经典异常检测算法库”

- 目标域更泛（tabular/embedding 多）
- 优点：成熟、稳定、sklearn 风格、生态强
- 不足：对视觉工业链路（像素定位、缺陷区域、对齐/模板检验）不是重点

PyImgAno 的定位更偏“视觉工业异常检测工具箱”：
- 除了 `core_*`/`vision_*` detector，还把 **输入/预处理/tiling/defects 输出/JSONL 工业工艺** 放进同一条链路里。

### anomalib：更像“训练/评估框架 + 视觉 AD model zoo”

anomalib 本质是一个框架（训练循环、数据模块、模型实现、导出/推理工具等），天然就是“复合包”。  
把 anomalib “拆开成一堆小包”在工程上很难，也会破坏其统一训练/评估体验。

PyImgAno 的推荐策略不是拆 anomalib，而是：

- 把 anomalib 作为 **可选 backend**：`pip install "pyimgano[anomalib]"`
- 在 PyImgAno 内提供 **wrapper/bridge**，让 anomalib 训练好的 checkpoint 能进入“工业推理/缺陷导出/JSONL”链路

对应实现：`pyimgano/models/anomalib_backend.py` 等  
对应文档：`docs/ANOMALIB_CHECKPOINTS.md`

---

## 3) 现阶段离“顶尖包/顶尖体验”还差什么？

下面是从“开发者/使用者视角”最常见的差距点（按影响优先级排序）：

### A. 更标准化的公开基准（可复现的 leaderboard）

顶尖包通常提供：
- 明确的 dataset splits（版本化）
- 一键复现实验配置（含 seeds、预处理、模型参数）
- 可直接引用的 benchmark 结果（表格/图）

PyImgAno 当前已经有 suite/sweep + artifacts，但还可以继续加强：
- ✅ 给出“官方基准配置集”（初版）：`benchmarks/configs/*.json` + `pyimgano-benchmark --config ...`
- ✅ suite 输出支持 `leaderboard.*` 导出（csv/md），run artifacts 包含 `config.json` + `environment.json`（含 git/包版本/安装的 extras）
- ⏳ 继续加强：版本化的 dataset splits + 更可引用的 leaderboard 元信息（环境哈希/依赖锁）

### B. 文档“最短路径”更短（减少选择焦虑）

虽然已经有 `QUICKSTART`/`CLI_REFERENCE`/`WORKBENCH`，但顶尖项目会更强调：
- 10 分钟走通最小闭环（demo → suite → infer → defects）
- 典型坑位（对齐、ROI、阈值、FP 调参）有非常具体的建议与示例命令

（PyImgAno 已有大量工业文档，但可以进一步把“最佳实践”收敛成更短的入口页。）

### C. 生态与社区（“别人愿不愿意贡献”）

顶尖包的优势往往来自：
- issue/PR 模板、贡献指南、清晰的架构与约束
- 稳定 API/semver 预期
- 快速的 CI 反馈与回归防护

PyImgAno 已有 CI、模板、贡献指南，但长期仍需要靠迭代积累。

### D. 预训练权重/模型卡（可审计 + 可复现）

视觉 AD 的“顶尖体验”通常会提供：
- 明确的权重来源、许可证、训练配置、适用范围（model card）
- 离线缓存与校验（hash/manifest）

PyImgAno 已强调离线安全与权重策略，但“官方权重库/模型卡”仍可继续建设。
目前已补齐基础设施（但不包含任何官方权重资产）：

- ✅ `pyimgano-weights`：本地 weights/checkpoints 的 sha256 计算 + manifest 校验（不下载任何东西）
- ✅ `docs/WEIGHTS.md`：权重策略 + manifest 使用方式
- ✅ `docs/MODEL_CARDS.md`：模型卡模板（方便企业内部资产化）

---

## 4) 推荐使用路径（给第一次上手的开发者）

### 路径 1：完全离线 CPU 快速验证（1 分钟）

```bash
pyimgano-demo
```

### 路径 2：工业算法选择（suite + 小网格扫参）

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-v4 \
  --suite-sweep industrial-feature-small \
  --suite-sweep-max-variants 1 \
  --no-pretrained \
  --device cpu
```

### 路径 3：生产导向闭环（Workbench → infer_config → infer → defects）

- `pyimgano-train ... --export-infer-config`
- `pyimgano-infer --infer-config ... --defects --save-jsonl ... --save-masks ...`

对应文档：`docs/WORKBENCH.md`、`docs/INDUSTRIAL_INFERENCE.md`

---

## 5) 下一步“最值”的升级建议（建议优先做 A）

如果你的目标是“让开发者一眼觉得这是顶尖包”，我建议按这个顺序：

1) **A（最优先）**：把“最短上手路径 + 工业最佳实践”进一步收敛（更短、更具体、更少选择）
2) **基准可复现**：为主流数据集维护“官方 suite 配置集合 + 结果导出模板”
3) **权重/模型卡**：把深度模型的 checkpoint 资产化（来源、哈希、许可、适用范围）
4) **更多工业 baselines**：继续补齐“对齐/模板/纹理/颜色/频域”方向的强 baseline，并配套 sweep
