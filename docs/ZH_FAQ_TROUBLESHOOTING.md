# pyimgano 中文 FAQ 与 Troubleshooting

这份文档面向已经开始使用 `pyimgano` 的工程同学。  
它不是从 0 到 1 的入门指南，而是一份更接近 GitHub / PyTorch 文档风格的中文排障与问答文档。

适用场景：

- 你已经能跑命令，但结果不稳定、误报偏高或产物不完整
- 你已经进入真实项目，需要更快定位问题而不是重新通读所有文档
- 你需要一份可以被团队共享、按问题直接检索的参考文档

如果你还没跑通基础链路，先看：

- `docs/ZH_ZERO_TO_ONE_GUIDE.md`

如果你已经在做行业选型和场景落地，再配合看：

- `docs/ZH_INDUSTRY_SCENARIO_PLAYBOOK.md`

## 1. 快速分诊

先不要一上来怀疑模型。  
大部分真实项目的问题，优先级通常是：环境和输入格式 > 数据契约 > 推理配置 > 后处理 > 模型本身。

| 现象 | 第一条命令 | 最常见原因 | 优先看哪份文档 |
|------|------------|------------|----------------|
| 命令跑不起来 / 模型缺依赖 | `pyimgano-doctor --json` | extras 没装全、环境缺口 | `docs/ZH_ZERO_TO_ONE_GUIDE.md` |
| `preflight` 有报错 | `pyimgano-train --config ./cfg.json --preflight` | 路径错、split 非法、group 泄漏、mask 缺失 | `docs/MANIFEST_DATASET.md` |
| `--defects` 开了但没有可用 mask | 看模型是否有 `pixel_map` 能力 | 选了 score-only 模型 | `docs/ZH_ZERO_TO_ONE_GUIDE.md` |
| 误报很多 | `--save-overlays` | ROI、边界、光照、tiling seam | `docs/FALSE_POSITIVE_DEBUGGING.md` |
| 大图结果异常、边缘有缝 | `--tile-size 512 --tile-stride 384 --tile-map-reduce hann` | overlap 不足、blending 不稳 | `docs/FALSE_POSITIVE_DEBUGGING.md` |
| 离线还行，现场变差 | `pyimgano-validate-infer-config ...` | 输入格式变化、现场条件漂移、配置未同步 | 本文第 5 节 |
| 交付后别人跑不起来 | 校验 `infer_config.json` / `deploy_bundle` | 交的是训练配置，不是部署配置 | `docs/ZH_ZERO_TO_ONE_GUIDE.md` |

## 2. 环境与安装 FAQ

### 为什么有些模型或 suite 看起来存在，但实际跑不起来？

最常见的原因是可选依赖没有装齐。  
`pyimgano` 里不少模型和能力是按 extras 管理的，不是默认全装。

先跑：

```bash
pyimgano-doctor --json
pyimgano-doctor --suite industrial-v4 --json
```

优先看：

- 你要用的 extras 是否可用
- 对应 suite 里的 baseline 是否被 skip
- Python、torch、onnxruntime、openvino 是否真的被当前环境识别

### 为什么我明明装了 GPU 环境，结果还是走 CPU？

先不要猜，先看环境：

```bash
pyimgano-doctor --accelerators --json
```

然后确认三件事：

- 当前环境里的 `torch` 是否带 CUDA
- 你在配置里是否真的指定了 `device: "cuda"`
- 部署机器和训练机器是不是同一个环境

### 什么情况下应该优先修环境，而不是继续调配置？

如果你已经出现下面任意一种情况，就应该先停下来修环境：

- `doctor` 已经提示关键 extras 不可用
- 你要的模型在当前环境里根本不能初始化
- 推理或训练命令一开始就因为依赖失败退出

这种时候继续调配置，通常只是浪费时间。

## 3. 数据与 Manifest FAQ

### 什么时候该从目录结构切到 manifest？

只要你开始遇到下面任意一个需求，就建议切 manifest：

- 多类目
- 多视角
- 多工位
- 多班次 / 多光照条件
- 需要防止同件产品泄漏到 train/test
- 需要长期维护和复盘

### `group_id` 到底解决什么问题？

`group_id` 的作用不是“多一个字段”，而是防止同一物理实体的数据同时出现在 train 和 test。  
如果一个产品有多视角、多张连拍、来自同一短视频片段，它们通常都应该共享一个 `group_id`。

如果这个字段没有处理好，离线结果很容易虚高。

### manifest 里哪些字段最值得优先保留？

最低建议保留：

- `image_path`
- `category`
- `split`
- `label`

真实项目里很有价值的扩展字段通常是：

- `group_id`
- `mask_path`
- `meta.view_id`
- `meta.station_id`
- `meta.condition`
- `meta.lot_id`
- `meta.template_id`

### `mask_path` 不是必须有，那我什么时候一定要补？

如果你关心下面这些指标或验收项，就应该尽量补：

- pixel AUROC
- pixel AP
- AUPRO
- mask 级定位质量

如果暂时没有 pixel 级 GT，也能做第一轮上线准备。  
但这时你更要依赖：

- hard negative 集合
- overlay 抽检
- 固定误报预算
- 业务对 region 形态的确认

### 相对路径和绝对路径，怎么选？

从迁移和交付角度，优先推荐相对路径。  
如果 manifest 要跨机器复制、打包进仓库、交给其他团队，相对路径明显更稳。

## 4. 训练与 Workbench FAQ

### `--dry-run` 和 `--preflight` 有什么区别？

可以把它们理解成两个不同层级的检查：

- `--dry-run` 先看配置能不能被解析
- `--preflight` 再看数据和能力契约是否真的成立

更稳的顺序是：

```bash
pyimgano-train --config ./cfg_manifest.json --dry-run
pyimgano-train --config ./cfg_manifest.json --preflight
```

### `preflight` 里哪些问题属于阻塞项？

下面这些问题通常不应该带着继续训练：

- 路径缺失
- split 非法
- `test` 标签不完整
- mask 覆盖异常
- `group_id` 泄漏
- 配置要求和模型能力不匹配

### 为什么推荐 `pyimgano-train --export-infer-config`？

因为真实项目最终交付的重点不是“你当时怎么训练的”，而是“别人怎么稳定推理”。  
`infer_config.json` 更接近部署契约，而不是训练过程记录。

推荐命令：

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config
```

### 什么时候应该直接导出 `deploy_bundle`？

当你准备把结果交给下面这些目标时，`deploy_bundle` 更稳：

- 另一台机器
- 容器镜像
- 服务端推理任务
- 别的团队

因为它更像一个最小可搬运推理包，而不是完整训练目录。

## 5. 推理与部署 FAQ

### 我应该直接用 `--model`，还是用 `--infer-config`？

结论：

- 做快速试验、模板法探路时，`--model` 很方便
- 进入真实项目交付后，优先 `--infer-config`

原因：

- `--model` 更适合探索
- `--infer-config` 更适合复现、审计、上线和 handoff

### 为什么推荐先校验 `infer_config.json`？

因为它能提前发现很多“看起来像模型问题，实际是路径或配置问题”的故障。

命令：

```bash
pyimgano-validate-infer-config /path/to/infer_config.json
```

适合在这些时刻执行：

- 训练刚导出之后
- 配置刚被复制到新机器之后
- 第一次进线之前

### 离线效果还行，为什么现场一接就变差？

优先排查这几类偏差：

- 输入尺寸不同
- 颜色格式不同
- 位深不同
- 现场新增了训练中没见过的视角、工位、模板或班次
- ROI / tiling / preprocessing 规则没有同步

最稳的现场 smoke run 一般是：

```bash
pyimgano-validate-infer-config /path/to/infer_config.json

pyimgano-infer \
  --infer-config /path/to/infer_config.json \
  --input /path/to/onsite_samples \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

### 12-bit / 16-bit 灰度图最容易踩什么坑？

最常见的问题不是模型，而是输入规范化没对齐。  
如果你的工业相机是 12-bit 数据存到 `uint16` 容器里，就要明确传入正确的 `u16_max`。

如果这一步没对齐，现场和离线的分布很容易完全不是一回事。

## 6. Defects / Overlay FAQ

### 为什么开了 `--defects`，结果还是不好用？

`--defects` 只负责把 anomaly map 变成 mask 和 regions。  
它不会自动修复前面的输入、模型、ROI 或 tiling 问题。

最常见的根因有：

- 模型本身没有稳定的 `pixel_map`
- ROI 没设对
- 图太大但没有 tiling
- 后处理参数不适合真实缺陷形态
- 根本没看 overlay

### 为什么我有 image score，但没有靠谱的缺陷 mask？

因为 image score 和 pixel-level 定位不是一回事。  
如果模型是 score-only，你可能能得到异常分数，但拿不到真正可用的 defects 输出。

更稳的第一选择通常是带 `pixel_map` 的模型，例如：

- `vision_patchcore`
- `vision_softpatch`
- `vision_padim`
- `ssim_template_map`
- `vision_pixel_mad_map`

### 误报很多时，第一步应该调什么？

不是 threshold。  
更稳的顺序通常是：

1. 先开 `--save-overlays`
2. 再看 ROI
3. 再看 border ignore
4. 再看 smoothing / hysteresis
5. 最后才去看 threshold

推荐起步命令：

```bash
pyimgano-infer \
  --infer-config /path/to/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-overlays /tmp/overlays \
  --save-masks /tmp/masks
```

### 为什么 tile seam 会像假缺陷？

最常见的原因是：

- 没有 overlap
- overlap 太小
- blending 方式过于生硬

先试：

```bash
--tile-size 512 \
--tile-stride 384 \
--tile-map-reduce hann
```

### 为什么线状缺陷总被过滤掉？

很可能是后处理规则不适合这类缺陷，比如：

- `min_area` 太大
- `max_aspect_ratio` 太小
- 形状过滤规则按圆斑噪点设计，却被拿去过滤裂纹或焊缝

长条、线状、边缘型缺陷项目里，不要直接照搬“去掉长细误报”的经验值。

## 7. 复制就能用的排障命令

### 环境检查

```bash
pyimgano-doctor --json
pyimgano-doctor --suite industrial-v4 --json
```

### 配置与数据契约检查

```bash
pyimgano-train --config ./cfg_manifest.json --dry-run
pyimgano-train --config ./cfg_manifest.json --preflight
```

### 部署配置校验

```bash
pyimgano-validate-infer-config /path/to/infer_config.json
```

### 部署式推理

```bash
pyimgano-infer \
  --infer-config /path/to/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/results.jsonl
```

### 带 defects 和 overlay 的排障推理

```bash
pyimgano-infer \
  --infer-config /path/to/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

## 8. 问题到动作的对照表

| 问题 | 先看什么 | 先做什么 |
|------|----------|----------|
| 环境不完整 | `doctor` 输出 | 补齐 extras |
| 配置跑不通 | `--dry-run` / `--preflight` | 修配置、修路径、修 split |
| 离线高分但线上不稳 | 输入格式、`infer_config.json` | 先做现场 smoke run |
| 没有 defects 输出 | 模型是否支持 `pixel_map` | 换模型或换路线 |
| 误报很多 | overlay | 先做 ROI 和后处理排查 |
| seam 明显 | tiling 参数 | 增 overlap，换 `hann` |
| 长条缺陷漏掉 | shape filters | 放宽长宽比和面积过滤 |
| 别人接不起来 | 交付物是否完整 | 优先交 `deploy_bundle` / `infer_config.json` |

## 9. 相关文档

建议按下面顺序继续看：

- `docs/ZH_ZERO_TO_ONE_GUIDE.md`
- `docs/ZH_INDUSTRY_SCENARIO_PLAYBOOK.md`
- `docs/INDUSTRIAL_QUICKPATH.md`
- `docs/MANIFEST_DATASET.md`
- `docs/FALSE_POSITIVE_DEBUGGING.md`
- `docs/CLI_REFERENCE.md`
