# pyimgano 从 0 到 1 操作指南

这份文档面向第一次把 `pyimgano` 用起来的工程同学，目标不是解释所有能力，而是给你一条 **从环境准备、数据落盘、训练导出、推理落地到部署交付** 的最短可执行路径。

适用场景：

- 你要在一个新的工业质检项目里落地异常检测
- 你希望产出可复现的训练记录和可部署的推理配置
- 你希望优先跑通一条稳妥路径，而不是一开始就试很多模型

如果你只想快速感受一下工具链，先看 `docs/QUICKSTART.md`。  
如果你已经进入真实项目，建议直接按本文走。

## 1. 先决定你走哪条数据路径

`pyimgano` 支持多种数据组织方式，但从工程角度只建议两条：

### 路径 A：标准目录结构

适合单类目、小规模、快速试跑。

目录示例：

```text
dataset_root/
  train/
    normal/
      *.png
  test/
    normal/
      *.png
    anomaly/
      *.png
```

优点：

- 最容易上手
- 可以直接用于 demo、benchmark 和简单训练

限制：

- 不适合多来源、多类目、复杂分组约束
- 不适合长期维护和跨团队协作

### 路径 B：manifest 数据集

适合真实项目，推荐作为默认方案。

manifest 是一个 JSONL 文件，每行描述一张图：

```jsonl
{"image_path":"images/train_0001.png","category":"bottle","split":"train","label":0}
{"image_path":"images/test_0101.png","category":"bottle","split":"test","label":1,"mask_path":"masks/test_0101.png"}
```

优点：

- 支持多类目
- 支持 `group_id` 防止数据泄漏
- 支持显式 mask 和元数据
- 更适合生产仓库、标注平台和批处理链路

详细格式见：`docs/MANIFEST_DATASET.md`

## 2. 安装最小依赖并检查环境

先按你的目标安装依赖：

```bash
pip install pyimgano
pip install "pyimgano[torch]"
```

如果你确定会用 ONNX / OpenVINO / scikit-image，再按需补装：

```bash
pip install "pyimgano[onnx]"
pip install "pyimgano[openvino]"
pip install "pyimgano[skimage]"
```

然后做一次环境自检：

```bash
pyimgano-doctor
pyimgano-doctor --json
pyimgano-doctor --suite industrial-v4 --json
```

你要重点看三件事：

- Python 环境和 `pyimgano` 版本是否正常
- 计划使用的 extras 是否真的可用
- 某个 suite 中哪些 baseline 会被跳过

如果你准备上 GPU，还可以额外看 accelerator：

```bash
pyimgano-doctor --accelerators --json
```

## 3. 先跑一个最小闭环，确认工具链没问题

如果你还没准备真实数据，先跑官方 demo：

```bash
pyimgano-demo
```

这个命令会自动：

- 生成一个极小的 `custom` 数据集
- 跑一套基线
- 导出报告和表格

这一步的目标不是看指标，而是确认：

- CLI 能正常运行
- 依赖没有明显缺失
- 输出目录结构和产物格式符合预期

如果你已经有真实数据，也建议先在极小子集上做 smoke run，而不是直接全量训练。

## 4. 为真实项目准备配置文件

最稳妥的方式是从现成模板拷贝：

```bash
cp examples/configs/manifest_industrial_workflow_balanced.json ./cfg_manifest.json
```

或者根据场景选别的模板：

- `examples/configs/industrial_adapt_fast.json`
- `examples/configs/industrial_adapt_preprocessing_illumination.json`
- `examples/configs/industrial_adapt_maps_tiling.json`
- `examples/configs/industrial_adapt_defects_roi.json`

你至少要改这些字段：

- `dataset.root`
- `dataset.manifest_path`（如果是 manifest）
- `dataset.category`
- `model.device`
- `output.output_dir`（可选，不写就落到 `runs/`）

如果你不确定模型该怎么选，建议先用下面的顺序：

1. 像素定位优先：`vision_patchcore`
2. 噪声 normal 较多：`vision_softpatch`
3. 想更轻量：`vision_padim`
4. 只有 CPU、先求快跑通：`vision_ecod`

## 5. 训练前先做 preflight 和 dry-run

不要直接开跑。先验证配置和数据契约。

### 只检查配置能否解析

```bash
pyimgano-train --config ./cfg_manifest.json --dry-run
```

### 检查数据是否存在阻塞问题

```bash
pyimgano-train --config ./cfg_manifest.json --preflight
```

你要重点看：

- train/test 数量是否合理
- mask 覆盖是否符合预期
- 是否有缺文件、坏路径、非法 split
- 是否有分组泄漏风险

如果这一步不干净，后面的训练和推理产物通常也不可信。

## 6. 跑训练并导出部署用推理配置

推荐命令：

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config
```

训练完成后，run 目录里最重要的是这些文件：

- `report.json`
- `config.json`
- `environment.json`
- `categories/<category>/per_image.jsonl`
- `artifacts/infer_config.json`

其中：

- `report.json` 是这次 run 的汇总结果
- `per_image.jsonl` 是逐图记录，适合审计和误报排查
- `infer_config.json` 是后续推理最关键的交付物

如果你要把产物交给别的机器、容器或流水线，建议直接导出 deploy bundle：

```bash
pyimgano-train --config ./cfg_manifest.json --export-deploy-bundle
```

这样你得到的是一个更适合交付的目录，而不是整个训练目录。

## 7. 校验导出的 infer-config

在做正式推理前，先校验导出的配置：

```bash
pyimgano-validate-infer-config runs/<run_dir>/artifacts/infer_config.json
```

如果你导出的是 deploy bundle：

```bash
pyimgano-validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json
```

这一步的目标是确认：

- schema 合法
- 依赖的 checkpoint / artifact 路径可解析
- 配置能在后续推理阶段被稳定消费

## 8. 用导出的 infer-config 做部署式推理

推荐不要再拿训练配置直接做线上推理，而是统一走：

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

如果 infer-config 里包含多个类目，再补一个：

```bash
--infer-category bottle
```

如果你要从完整 run 直接复用，也可以：

```bash
pyimgano-infer --from-run /path/to/run_dir --input /path/to/images
```

但从生产交付角度，还是更推荐 `--infer-config`。

## 9. 需要缺陷 mask / region 时怎么做

如果你的下游系统不只是要一个异常分数，而是要：

- 二值缺陷 mask
- 连通域框
- 每个缺陷区域的面积 / bbox / score

那就启用 defects：

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/pyimgano_results.jsonl \
  --save-masks /tmp/pyimgano_masks
```

如果训练阶段已经把缺陷阈值、ROI、形态学参数写进了 `infer_config.json`，这里会直接复用；CLI 显式传入的参数仍然优先生效。

需要更完整的像素级推理说明时，看：

- `docs/INDUSTRIAL_INFERENCE.md`
- `docs/FALSE_POSITIVE_DEBUGGING.md`

## 10. 需要做算法选型时怎么跑

不要一上来自己手工挑 10 个模型乱跑。先用 suite：

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/dataset_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --suite industrial-v4 \
  --device cpu \
  --no-pretrained \
  --output-dir /tmp/pyimgano_suite_run
```

如果想给每个 baseline 再试一点小变体，可以加 sweep：

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/dataset_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --suite industrial-v4 \
  --suite-sweep industrial-template-small \
  --suite-sweep-max-variants 1 \
  --suite-export csv \
  --output-dir /tmp/pyimgano_suite_sweep_run
```

你最后应该拿这个结果做决策：

- 先看是否有稳定可跑的 baseline
- 再看 image-level 和 pixel-level 指标
- 最后才看更复杂或更重的模型要不要继续深挖

## 11. 什么时候该用 numpy-first 推理

如果你的图像不是从磁盘路径进来，而是已经在内存里，比如：

- OpenCV 视频帧
- 相机 SDK 回调
- 自己的服务接口传 numpy 数组

那就应该走 numpy-first 推理接口，而不是硬落盘再读回来。

相关能力见：

- `docs/INDUSTRIAL_INFERENCE.md`

核心原则：

- 明确传入 `ImageFormat`
- 统一归一化成 `RGB / uint8 / HWC`
- 需要大图时再加 tiling

## 12. 交付上线前建议保留哪些产物

一个最小可追溯的交付包，建议至少包含：

- `infer_config.json`
- 对应 checkpoint 或 deploy bundle
- 一份 `report.json`
- 一份 `environment.json`
- 一小批真实样本推理生成的 `results.jsonl`

这样你后面排查问题时，至少能回答：

- 当时用的是什么模型和阈值
- 环境是不是变了
- 推理输出格式是否一致
- 某张图在当时到底打了多少分

## 13. 常见起步建议

如果你完全从零开始，不想纠结，直接按下面做：

1. 安装 `pyimgano[torch]`
2. 用 manifest 组织数据
3. 用 `vision_patchcore` 起步
4. 先跑 `--preflight`
5. 再跑 `pyimgano-train --export-infer-config`
6. 用 `pyimgano-infer --infer-config ...` 做推理
7. 需要缺陷导出时再打开 `--defects`
8. 需要选型时再跑 `industrial-v4` suite

这条路径不是最花哨的，但通常是最稳的。

## 14. 进一步阅读

- `docs/WORKBENCH.md`
- `docs/MANIFEST_DATASET.md`
- `docs/INDUSTRIAL_INFERENCE.md`
- `docs/CLI_REFERENCE.md`
- `docs/ROBUSTNESS_BENCHMARK.md`
- `docs/FALSE_POSITIVE_DEBUGGING.md`
- `docs/ALGORITHM_SELECTION_GUIDE.md`
- `docs/ZH_INDUSTRY_SCENARIO_PLAYBOOK.md`
- `docs/ZH_FAQ_TROUBLESHOOTING.md`

## 15. 最常见的 8 类问题，先按这个顺序排查

第一次落地时，真正浪费时间的通常不是模型本身，而是环境、数据契约和推理参数。建议按下面顺序排查。

### 1) `doctor` 先看 extras 是否齐

如果你发现某个模型、suite 或导出功能跑不起来，先不要直接翻代码，先看环境：

```bash
pyimgano-doctor --json
pyimgano-doctor --suite industrial-v4 --json
```

优先确认：

- `torch` / `torchvision` 是否可用
- `onnxruntime` / `openvino` 是否真的安装成功
- 你想跑的 suite 是否因为 extras 缺失而被跳过

### 2) `preflight` 不干净，不要继续训练

如果 `pyimgano-train --preflight` 里已经出现：

- 缺文件
- split 非法
- group 泄漏
- mask 覆盖异常
- 模型和预处理能力不匹配

那就应该先修数据或配置，而不是继续开跑。

### 3) 模型不支持 numpy，但你又开了 numpy-first 能力

有些能力天然要求模型支持 numpy 输入，比如：

- 某些预处理链
- robustness corruptions
- 高分辨率 tiling

如果你打开这些能力，却选了不支持 numpy 的模型，常见结果是 preflight 报错，或者推理时能力不匹配。

处理方式通常只有两个：

- 换成支持 `numpy` tag 的模型
- 关掉依赖 numpy-first 的功能

### 4) 开了 `--defects`，但模型没有 anomaly map

`--defects` 依赖像素级 anomaly map。  
如果模型本身是 score-only，通常不会有可用的缺陷 mask / region。

遇到这种情况，不要硬调阈值，直接换成带 `pixel_map` 能力的模型，比如：

- `vision_patchcore`
- `vision_softpatch`
- `vision_padim`

### 5) 测试集里 anomaly 没有 mask，像素指标会不可信

如果你要算 pixel-level 指标，但测试异常样本没有对应 mask，结果通常会跳过或失真。

这时要先确认：

- manifest 里是否真的写了 `mask_path`
- `mask_path` 路径能否解析
- mask 是否和 image 一一对应

### 6) 默认是离线安全，不会自动帮你下载权重

`pyimgano` 的 CLI 默认更偏离线安全，很多地方相当于默认 `--no-pretrained`。

如果你用的模型依赖上游权重，而你又没显式开启，常见现象是：

- 模型初始化失败
- 要求你手动指定 embedder / checkpoint

这不是 bug，通常是为了防止在生产或 CI 里静默下载权重。

### 7) 大批量推理中单张坏图会拖死整批

如果你一次推大量图片，建议别只跑最朴素模式。更稳妥的方式是：

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/results.jsonl \
  --continue-on-error \
  --max-errors 100 \
  --flush-every 100
```

这样即使某张图损坏，也不会让整批结果全丢。

### 8) 误报太多，先看 overlay，不要先拍脑袋调阈值

误报问题建议按这个顺序处理：

1. 先开 `--save-overlays`
2. 再看 ROI 是否需要裁掉背景
3. 再看 border / smoothing / hysteresis / min-area
4. 最后再动阈值

对应文档：`docs/FALSE_POSITIVE_DEBUGGING.md`

## 16. 批量推理和生产任务推荐这样跑

如果你要做班次批处理、夜间批跑、产线回放，建议不要只保留最短命令，而是加上稳定性参数。

推荐模板：

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images_or_dir \
  --save-jsonl /tmp/pyimgano_results.jsonl \
  --continue-on-error \
  --max-errors 100 \
  --flush-every 100 \
  --profile-json /tmp/pyimgano_profile.json
```

这些参数分别解决不同问题：

- `--continue-on-error`：单张失败时继续跑，JSONL 里记录错误
- `--max-errors`：错误数量过多时提前停，防止任务无限消耗资源
- `--flush-every`：减少进程中断时的结果丢失
- `--profile-json`：留下机器可读的耗时统计，方便做监控

如果你还要做缺陷导出和误报排查，可以再叠加：

```bash
--defects \
--save-masks /tmp/pyimgano_masks \
--save-overlays /tmp/pyimgano_overlays
```

## 17. 一个推荐的项目交付目录

如果你准备把结果交给部署同学、平台同学或别的仓库，建议整理成下面这种结构：

```text
handoff/
  infer_config.json
  report.json
  environment.json
  sample_results.jsonl
  sample_inputs/
    good_0001.png
    bad_0007.png
  sample_outputs/
    results.jsonl
    masks/
    overlays/
  checkpoints/
    ...
```

如果你直接用 `--export-deploy-bundle`，很多内容会自动按可部署布局整理好。  
你还需要自己补的，通常是：

- 几张代表性输入样本
- 对应输出结果
- 你们内部如何调用的命令模板

交付时建议写清楚 4 件事：

- 推理入口命令
- 是否需要 GPU
- 哪些环境变量要固定
- 结果文件输出到哪里

## 18. 一个推荐的命令清单

如果你接手一个新项目，通常会按下面顺序跑这些命令。

### 阶段 A：环境验证

```bash
pyimgano-doctor --json
pyimgano-doctor --suite industrial-v4 --json
```

### 阶段 B：数据验证

```bash
pyimgano-train --config ./cfg_manifest.json --dry-run
pyimgano-train --config ./cfg_manifest.json --preflight
```

### 阶段 C：第一次训练

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config
```

### 阶段 D：第一次部署式推理

```bash
pyimgano-validate-infer-config runs/<run_dir>/artifacts/infer_config.json

pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/results.jsonl
```

### 阶段 E：需要缺陷输出

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks
```

### 阶段 F：需要误报排查

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

### 阶段 G：需要算法选型

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/dataset_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --suite industrial-v4 \
  --device cpu \
  --no-pretrained \
  --output-dir /tmp/pyimgano_suite_run
```

## 19. 最后给第一次落地的建议

第一次用 `pyimgano`，最容易犯的错误有两个：

- 一开始就想把所有能力一起打开
- 还没把数据和产物链路跑通，就急着比较算法指标

更稳妥的顺序应该是：

1. 先让 `doctor` 和 `preflight` 干净
2. 再拿一个稳定基线导出 `infer_config.json`
3. 再验证部署式推理输出
4. 最后再做 defects、robustness、suite 选型和更重的模型

先把链路做稳，后面不管换模型、换阈值还是换部署形态，成本都会低很多。

## 20. 一个最小 manifest 配置样例

如果你现在就想照抄一份最小配置开始，下面这个版本最适合作为起点：

```json
{
  "recipe": "industrial-adapt",
  "seed": 123,
  "dataset": {
    "name": "manifest",
    "root": "/path/to/dataset_root",
    "manifest_path": "/path/to/manifest.jsonl",
    "category": "all",
    "resize": [256, 256],
    "input_mode": "paths",
    "split_policy": {
      "mode": "benchmark",
      "scope": "category",
      "seed": 123,
      "test_normal_fraction": 0.2
    }
  },
  "model": {
    "name": "vision_patchcore",
    "device": "cuda",
    "preset": "industrial-balanced",
    "pretrained": true,
    "contamination": 0.1
  },
  "adaptation": {
    "save_maps": true
  },
  "output": {
    "save_run": true,
    "per_image_jsonl": true
  }
}
```

这个配置适合：

- 你已经有 manifest
- 你需要 anomaly map
- 你希望后续能直接导出 `infer_config.json`

如果你还没有 GPU，把 `model.device` 改成 `cpu` 即可；只是速度会慢一些。

## 21. 一个带 defects 的配置样例

如果你的目标是直接落缺陷 mask / region，而不是只做图像级分数，可以从这个方向起步：

```json
{
  "recipe": "industrial-adapt",
  "seed": 123,
  "dataset": {
    "name": "mvtec",
    "root": "/path/to/mvtec_ad",
    "category": "bottle",
    "resize": [256, 256],
    "input_mode": "paths"
  },
  "model": {
    "name": "vision_patchcore",
    "device": "cuda",
    "preset": "industrial-balanced",
    "pretrained": true,
    "contamination": 0.1
  },
  "adaptation": {
    "save_maps": true
  },
  "defects": {
    "enabled": true,
    "pixel_threshold_strategy": "normal_pixel_quantile",
    "pixel_normal_quantile": 0.999,
    "mask_format": "png",
    "roi_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
    "border_ignore_px": 2,
    "map_smoothing": {
      "method": "median",
      "ksize": 3,
      "sigma": 0.0
    },
    "hysteresis": {
      "enabled": true,
      "low": null,
      "high": null
    },
    "shape_filters": {
      "min_fill_ratio": 0.15,
      "max_aspect_ratio": 6.0,
      "min_solidity": 0.8
    },
    "merge_nearby": {
      "enabled": true,
      "max_gap_px": 1
    },
    "min_area": 16,
    "min_score_max": 0.6,
    "min_score_mean": null,
    "open_ksize": 0,
    "close_ksize": 0,
    "fill_holes": false,
    "max_regions": 20,
    "max_regions_sort_by": "score_max"
  },
  "output": {
    "save_run": true,
    "per_image_jsonl": true
  }
}
```

这份配置不是“最好”的默认值，但它代表了一条比较典型的工业 FP 控制起点。  
如果后面发现误报还是高，优先去看：

- ROI 是否画对
- `pixel_normal_quantile` 是否过低
- `min_area` 是否过小
- `shape_filters` 是否太宽松

## 22. 推荐的团队分工方式

如果这个项目不是你一个人做，建议把职责切清楚，不要所有人同时改数据、模型和部署。

### 数据同学负责

- manifest 生成与维护
- `group_id` / `mask_path` / `label` 的正确性
- 训练集、测试集和漏标问题确认

### 算法同学负责

- 模型选择
- 配置调优
- `preflight` / `benchmark` / `train` / `infer` 主流程
- `report.json` 和 `per_image.jsonl` 解读

### 部署同学负责

- `deploy_bundle` 或 `infer_config.json` 的接入
- 线上环境变量、缓存目录和 GPU 资源规划
- 批量推理稳定性参数
- 输出文件接入下游系统

这么分的好处是：  
数据问题不会被误判成模型问题，部署问题也不会被误判成训练问题。

## 23. 上线前最后检查一次这 10 项

在你准备把这套东西交出去之前，建议手工过一遍下面这份清单：

1. `pyimgano-doctor --json` 是否没有明显环境缺口
2. `pyimgano-train --preflight` 是否没有阻塞级错误
3. 最终使用的配置文件是否已经固化保存
4. `report.json` 是否已归档
5. `environment.json` 是否已归档
6. `infer_config.json` 是否已校验
7. 是否保留了一小批样例输入和对应输出
8. 是否明确了推理命令和输出目录
9. 是否明确了是否需要 GPU / CUDA / 特定缓存目录
10. 是否明确了下游究竟消费 `score`、`label`、`mask` 还是 `regions`

如果这 10 项里有 3 项以上答不上来，通常说明现在更适合继续整理交付，而不是急着上线。
