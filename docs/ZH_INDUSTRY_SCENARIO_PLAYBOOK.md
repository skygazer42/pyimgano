# pyimgano 行业场景落地手册

这份文档不是通用入门，而是按常见工业场景拆分的落地建议。

如果你还没有跑通基本链路，先看：

- `docs/ZH_ZERO_TO_ONE_GUIDE.md`

如果你已经准备开始做真实项目选型，本文更适合作为“场景到方案”的参考。

## 1. 先看这张总表

| 场景 | 首选路径 | 起步模型 | 建议重点 |
|------|----------|----------|----------|
| 金属/玻璃/涂装表面划伤、脏污、压痕 | 像素定位优先 | `vision_patchcore` / `vision_softpatch` | ROI、误报抑制、像素级导出 |
| 标签、印刷、喷码、字符缺陷 | 对齐/模板优先 | `ssim_template_map` / `vision_phase_correlation_map` | 对齐、模板稳定、边界误报 |
| PCB、连接器、电子件外观 | 高分辨率 + 多视角 | `vision_patchcore` / `vision_softpatch` / `vision_anomalydino` | tiling、view 分组、细小缺陷 |
| 纺织、皮革、重复纹理表面 | 纹理建模优先 | `vision_padim` / `vision_patchcore` | 重复纹理、光照变化、mask 质量 |
| 电池极片、涂布、焊缝、长条结构缺陷 | ROI + 高分辨率 | `vision_patchcore` / `vision_pixel_mad_map` | 线状缺陷、长宽比过滤不能过强 |
| 少样本、紧急上线、normal 很少 | few-shot / 稳妥基线 | `vision_anomalydino` / `vision_patchcore` | 少样本、离线安全、快速交付 |

不要把这张表理解成“唯一正确答案”。  
它的作用是帮你缩小第一轮尝试范围，避免一上来横向试十几个模型。

## 2. 用这份手册的方式

建议你按下面顺序阅读每个场景：

1. 先看这个场景的数据特征是否像你
2. 再看推荐的第一批模型
3. 再看最容易踩的误区
4. 最后复制建议命令去跑第一轮

如果一个项目同时命中两个场景，比如“PCB + 多视角 + 小样本”，优先保留更强的约束条件：

- 多视角问题优先按多视角处理
- 少样本问题优先按 few-shot 处理
- 高分辨率问题优先按 tiling 处理

## 3. 场景一：金属、玻璃、涂装件表面划伤 / 脏污 / 压痕

### 典型特征

- 异常通常是局部的
- 你关心的不只是图像级判断，还关心 defect mask / region
- 背景、夹具、边框往往会带来强误报

### 推荐起步模型

第一批建议只试这几个：

- `vision_patchcore`
- `vision_softpatch`
- `vision_padim`

如果工位极其稳定、产品对齐非常好，也可以补一组模板型基线：

- `vision_pixel_mad_map`
- `ssim_template_map`
- `vision_phase_correlation_map`

### 推荐策略

- 先保证有 `pixel_map`
- 先开 `save_maps`
- 推理时优先启用 `--defects`
- 尽早加 ROI，不要用阈值硬抗背景误报

### 常见误区

- 没有 ROI 就开始调阈值
- 误报很多时先怀疑模型，而不是先看 overlay
- 细小缺陷场景里把 `min_area` 设得太大

### 推荐命令

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config

pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

## 4. 场景二：标签、印刷、喷码、字符缺陷

### 典型特征

- 真正的异常往往是很小的字符缺笔、重影、污点、偏移
- 视觉差异经常来自位置偏差，而不是内容本身
- 相机位姿稳定时，模板方法常常比通用 embedding 方法更直接

### 推荐起步模型

优先顺序建议是：

- `ssim_template_map`
- `vision_phase_correlation_map`
- `vision_pixel_mad_map`

如果工位不够稳定、字体变化更大，再补：

- `vision_patchcore`
- `vision_padim`

### 推荐策略

- 先验证对齐质量，再看算法
- 模板法跑不稳时，很多时候不是模型问题，而是位置漂移问题
- 字符区域很明确时，强烈建议加 ROI

### 常见误区

- 小字符场景直接拿大面积阈值规则
- 没有对齐就用模板差分
- 忽略边界和压缩噪声带来的假热点

### 推荐命令

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/good \
  --input /path/to/test/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks
```

如果有轻微位移，再试：

```bash
pyimgano-infer \
  --model vision_phase_correlation_map \
  --train-dir /path/to/train/good \
  --input /path/to/test/images \
  --include-maps \
  --defects
```

## 5. 场景三：PCB、连接器、电子件外观

### 典型特征

- 缺陷小、结构复杂、背景纹理多
- 常常有多视角、多工位、多相机
- 高分辨率图像很常见

### 推荐起步模型

- `vision_patchcore`
- `vision_softpatch`
- `vision_anomalydino`

其中：

- `vision_patchcore` 是最稳的默认起点
- `vision_softpatch` 更适合 normal 数据里混进脏样本
- `vision_anomalydino` 更适合 normal 很少、想要 few-shot 起步

### 数据组织建议

这类项目强烈建议用 manifest，并尽量补这些字段：

- `category`
- `group_id`
- `meta.view_id`
- `meta.condition`

这样后面你才能清楚地区分：

- 同一产品不同视角
- 不同相机
- 不同照明工位

### 推荐策略

- 高分辨率时优先上 tiling
- 多视角问题不要混成单视角普通项目处理
- 误报排查时一定保留 overlays

### 推荐命令

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --include-maps \
  --tile-size 512 \
  --tile-stride 384 \
  --tile-map-reduce hann \
  --save-jsonl /tmp/results.jsonl
```

## 6. 场景四：纺织、皮革、重复纹理表面

### 典型特征

- 正常样本里本身就有丰富纹理
- 异常有时不是形状突变，而是局部纹理统计变化
- 光照和布料褶皱很容易造成假阳性

### 推荐起步模型

- `vision_padim`
- `vision_patchcore`
- `vision_spade`

如果工位非常稳定，也可以补：

- `ssim_template_map`

### 推荐策略

- 先保证训练集 normal 的覆盖足够丰富
- 光照漂移明显时，优先做预处理和 robustness 检查
- 不要只盯 image-level 指标，要看 pixel-level 和实际 mask 形态

### 常见误区

- 用过少的 normal 就直接上线
- 只看 AUROC，不看实际缺陷区域是否碎裂
- 把皱折、阴影、光斑当成模型缺陷

### 推荐补充动作

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/root \
  --category carpet \
  --model vision_patchcore \
  --device cpu \
  --no-pretrained \
  --corruptions lighting,blur \
  --severities 1 2 3
```

## 7. 场景五：电池极片、涂布、焊缝、长条结构缺陷

### 典型特征

- 图像往往很大
- 缺陷是长条、细裂纹、边缘不齐、涂布不均
- ROI 很明确，但异常形态不一定是小圆斑

### 推荐起步模型

- `vision_patchcore`
- `vision_softpatch`
- `vision_pixel_mad_map`

如果工艺和对齐非常稳定，`vision_pixel_mad_map` 这类 per-pixel 基线会非常有价值，因为：

- 速度快
- 容易解释
- 适合稳定站位的长条结构检测

### 推荐策略

- 一开始就明确 ROI
- 高分辨率优先开 tiling
- 线状异常场景里，不要把 `max_aspect_ratio` 设得太小

### 常见误区

- 用针对圆斑噪点的 shape filter 去过滤真实裂纹
- 没有 ROI 就把整个背景一起送去做 defects
- 图太大但仍然按单张缩放到 224 或 256 去赌结果

### 推荐命令

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config

pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --tile-size 512 \
  --tile-stride 384 \
  --tile-map-reduce hann \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

## 8. 场景六：少样本、紧急上线、normal 很少

### 典型特征

- 只有很少正常样本
- 交付压力大，没有足够时间做多轮选型
- 你更关心先跑通和先交付，而不是一次达到最优

### 推荐起步模型

- `vision_anomalydino`
- `vision_patchcore`
- 如果工位稳定，再补 `ssim_template_map`

### 推荐策略

- 先求一个稳定可解释的基线
- 少样本条件下不要同时引入太多可调参数
- 明确区分“离线评估效果”与“上线稳定性”

### 最稳的一条路径

1. `doctor`
2. `preflight`
3. `vision_patchcore` 或 `vision_anomalydino`
4. `export-infer-config`
5. `infer-config` 推理
6. 产出样例输入和样例输出

如果这一步你还不确定先跑 native 基线还是参考 upstream 包装器，可以先让 `doctor` 按目标给一份候选：

```bash
pyimgano-doctor \
  --dataset-target /path/to/dataset \
  --objective latency \
  --allow-upstream native-only \
  --topk 2 \
  --json
```

返回结果里会同时给出：

- `selection_context`：这次选型是按什么目标筛的
- `candidate_pool_summary`：候选池里 native / wrapper 的数量
- `rejected_candidates`：哪些候选因为 upstream 策略或 extras 缺失被过滤掉

如果你想做“native vs upstream wrapper”的并排参考，可以直接跑：

```bash
pyimgano-benchmark \
  --dataset custom \
  --root /path/to/dataset \
  --category default \
  --suite industrial-parity-v1 \
  --device cpu
```

这个 suite 会同时带上：

- 轻量模板基线
- 轻量结构分数基线
- `industrial-embedding-core-balanced`
- native `vision_patchcore`
- `vision_patchcore_anomalib`
- `vision_patchcore_inspection_checkpoint`

如果你接的是 `patchcore-inspection` 已训练好的 saved-model 目录，交付前也可以直接审一下 bundle：

```bash
pyimgano-doctor --deploy-bundle /path/to/deploy_bundle --json
```

结果里会额外出现 `external_checkpoint_audit`，用来提示：

- 当前 artifact 是否被识别成 `patchcore-saved-model`
- 关键文件是否齐全
- 该 checkpoint 是否属于版本敏感型包装器

### 不建议一开始就做的事

- 同时比较十几个模型
- 一开始就做复杂集成
- 一开始就假设必须上最重的 foundation model

### 推荐命令

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config

pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/results.jsonl
```

## 9. 什么时候该优先模板法，什么时候该优先 embedding 法

### 模板法优先

满足下面条件时，优先试模板型基线：

- 工位稳定
- 产品姿态稳定
- 参考图容易获取
- 你最关心的是局部差异

常见模型：

- `ssim_template_map`
- `vision_phase_correlation_map`
- `vision_pixel_mad_map`

### embedding 法优先

满足下面条件时，优先试 embedding/memory-bank 路线：

- 正常样本存在自然变化
- 背景和局部纹理更复杂
- 你不指望一张模板图解决问题
- 你需要更泛化的局部异常建模

常见模型：

- `vision_patchcore`
- `vision_softpatch`
- `vision_padim`
- `vision_anomalydino`

## 10. 给每个场景都适用的 5 条硬建议

1. 第一次落地时，不要同时开太多能力。
2. 需要缺陷导出时，优先选择 `pixel_map` 模型。
3. 需要多视角、多工位或复杂元数据时，优先使用 manifest。
4. 误报排查时，永远先看 overlay，再改阈值。
5. 要交付部署时，优先交 `infer_config.json` 或 `deploy_bundle`，不要只交训练配置。

## 11. 场景化数据采集建议

### 金属、玻璃、涂装件表面

- `normal` 不要只采最干净、最平整的样本，要覆盖轻微反光差异、边缘位置变化和常见夹具入镜情况。
- 首轮就准备一批“难负样本”，比如强反光、边框高亮、可接受污点、轻微灰尘，它们通常决定误报上限。
- 如果真异常很少，先把最容易误报的 20 到 50 张 `normal` 单独留作验收集，比盲目找少量 defect 更有价值。

### 标签、印刷、喷码、字符

- 每种版式、字体、喷码密度和墨色状态都至少采一批 `normal`，不要默认一张模板图能覆盖所有变化。
- 数据里要保留轻微位置漂移、轻微拉伸、轻微模糊这类“像异常但其实可放行”的样本。
- 如果生产线上会换卷材、换班次、换打印头，采集时要把这些条件拆开记录，后面更容易定位误报来源。

### PCB、连接器、电子件

- 强烈建议按 `view_id`、`camera_id`、`condition` 组织数据，至少保证同一件产品不同视角能用 `group_id` 关联起来。
- 不要把高分辨率图先粗暴缩小再采样验证，小缺陷项目里，原始分辨率是否保留经常直接决定上限。
- 首轮验收集要专门保留“容易混淆但应判正常”的样本，例如丝印差异、焊点颜色变化、轻微反光。

### 纺织、皮革、重复纹理

- `normal` 必须覆盖不同卷料、批次、纹理朝向、张力和光照，不然模型学到的是某一批布，而不是“正常纹理”。
- 把褶皱、阴影、亮斑、折痕这些强干扰样本单独留一份 hard negative 集合，后面调参时优先看它。
- 如果现场会有明显照度变化，采集时不要只留静态最佳工况，否则上线后误报会很快失控。

### 电池极片、涂布、焊缝、长条结构

- 采集时优先保留原始长宽比和原始分辨率，不要为了省事先裁成统一小图。
- 要覆盖头尾段、拼接段、边缘段和最容易出现工艺波动的位置，因为这些位置最容易把 ROI 和 defects 规则打穿。
- 如果现场支持，最好额外保留一批“无缺陷但纹理波动大”的长条图，用来验证长连通区域误报。

### 少样本、紧急上线

- 少样本不等于随便拿几张图就开跑，优先争取一批拍摄条件稳定、命名可追溯的 `normal`，哪怕数量不多也要干净。
- 至少单独留出一小批 hardest normal 作为上线前守门集，不要把所有样本都拿去训练。
- 如果几乎没有 defect，可以先按“误报不能太多、结果要可解释、产物可交付”这三条目标组织验收样本。

## 12. 场景化验收口径建议

| 场景 | 首看指标 | 上线前至少确认 | 不要只看 |
|------|----------|----------------|----------|
| 金属/玻璃/涂装表面 | `pixel AUROC`、`pixel AP`、`AUPRO`、缺陷区域召回 | 固定误报预算下的 defect recall，overlay 是否真能圈住缺陷 | 只看 image AUROC |
| 标签/印刷/喷码 | 字符 ROI 内的 region precision / recall，固定样本量下误报数 | 关键字符漏检率，边界噪声是否稳定被压住 | 只看全图像素指标 |
| PCB/连接器/电子件 | 小缺陷召回、按 `view_id` 分组后的最差视角表现 | 多视角之间是否有数据泄漏，最差相机是否仍可接受 | 只看整体平均分 |
| 纺织/皮革/重复纹理 | `pixel AUROC`、`pixel_segf1`、hard negative 误报率 | 光照变化和褶皱条件下，阈值是否还能稳定工作 | 只看单一阈值下的一次结果 |
| 电池极片/涂布/焊缝 | 长连通区域是否完整，线状缺陷召回，边缘误报率 | 连续缺陷是否被切碎，头尾段和边缘段是否稳定 | 只看 `min_area` 调出来的漂亮结果 |
| 少样本/紧急上线 | 首轮可复现性、hard negative 通过率、结果可解释性 | `infer_config.json` 是否可交付，离线与上线输入是否一致 | 只看少量样本上的高分 |

补一句更接地气的话：

- 真正上线前，最好和业务约定一个固定误报预算，比如“每百张 normal 允许多少误报”或“每千张图最多多少次人工复检”。
- 如果项目要输出 mask / region，验收口径里一定要包含“框得对不对”和“是否碎裂”，不能只验 image score。

## 13. 按场景选起步模板

下面这张表的作用不是替你做最终选型，而是帮你把第一轮配置模板选对。

| 场景 | 建议先拷的模板 | 什么时候优先用它 |
|------|----------------|------------------|
| 金属/玻璃/涂装表面 | `examples/configs/industrial_adapt_defects_roi.json` | 你已经知道 ROI 明确，而且后面一定要导出 defects |
| 标签/印刷/喷码 | 直接先用 `pyimgano-infer --model ssim_template_map` | 工位稳定、模板法大概率比 workbench 训练更快出答案 |
| PCB/连接器/电子件 | `examples/configs/industrial_adapt_maps_tiling.json` | 高分辨率、小缺陷、多视角，第一轮先把 tiling 跑顺 |
| 纺织/皮革/重复纹理 | `examples/configs/industrial_adapt_preprocessing_illumination.json` | 你已经确认光照漂移是大问题，需要先压预处理和鲁棒性 |
| 电池极片/涂布/焊缝 | `examples/configs/industrial_adapt_maps_tiling.json` | 图很大、结构很长，先保住分辨率和 map 质量 |
| 少样本/紧急上线 | `examples/configs/manifest_industrial_adapt_fast.json` | 目标是先交付第一版，而不是一次把最优指标卷出来 |

如果你不知道从哪个模板起步，最保守的默认项是：

```bash
cp examples/configs/manifest_industrial_workflow_balanced.json ./cfg_manifest.json
```

然后按下面顺序改：

1. 改 `dataset.root`
2. 改 `dataset.manifest_path`
3. 改 `dataset.category`
4. 改 `model.device`
5. 如果知道场景强约束，再补 ROI / tiling / preprocessing

## 14. 按场景复制的首轮命令模板

### 表面缺陷、PCB、长条结构这三类

这三类通常都值得先走 workbench，再用导出的 `infer_config` 做部署式推理：

```bash
pyimgano-train --config ./cfg_manifest.json --preflight
pyimgano-train --config ./cfg_manifest.json --export-infer-config

pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks
```

如果图像分辨率很高，再补：

```bash
--tile-size 512 --tile-stride 384 --tile-map-reduce hann
```

### 标签、印刷、喷码

这类项目第一轮可以先不用训练，直接试模板型基线：

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/normal \
  --input /path/to/images \
  --defects-preset industrial-defects-fp40 \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

如果稳定性不足，再切到：

```bash
vision_phase_correlation_map
```

### 纺织、皮革、重复纹理

这类项目更适合先把 normal 覆盖做扎实，再跑一轮带预处理的 workbench：

```bash
cp examples/configs/industrial_adapt_preprocessing_illumination.json ./cfg_texture.json
pyimgano-train --config ./cfg_texture.json --export-infer-config
```

做完第一轮后，优先检查：

- hard negative 上是否稳定
- mask 是否碎裂
- 光照变化下阈值是否还能复用

### 少样本、紧急上线

这类项目第一目标是尽快交一个可复现的最小闭环：

```bash
cp examples/configs/manifest_industrial_adapt_fast.json ./cfg_fast.json
pyimgano-train --config ./cfg_fast.json --preflight
pyimgano-train --config ./cfg_fast.json --export-infer-config
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/results.jsonl
```

首轮完成后，不要急着上更多模型，先确认三件事：

- 结果能否复现
- 误报是否在可接受范围
- 导出产物是否真的能被下游系统消费

## 15. 按场景看最常见误报来源

这部分不是讲理论，而是帮你在第一次看 overlay 时更快判断问题出在哪。

| 场景 | 最常见误报来源 | 第一怀疑对象 | 第一轮优先动作 |
|------|----------------|--------------|----------------|
| 金属/玻璃/涂装表面 | 高光、边框反射、夹具、边缘阴影 | ROI 没收干净，边界像素太活跃 | 先加 ROI，再试 `border_ignore_px` 和 overlays |
| 标签/印刷/喷码 | 位置轻微漂移、模板不稳、边缘锯齿、压缩噪声 | 对齐问题，不是模型本身 | 先换更稳的对齐/模板，再看 threshold |
| PCB/连接器/电子件 | 丝印差异、针脚反光、焊点颜色波动、多视角混淆 | 数据分组和分辨率处理不对 | 先看 `view_id` / `group_id`，再看 tiling 和 overlays |
| 纺织/皮革/重复纹理 | 褶皱、阴影、方向变化、批次纹理差 | `normal` 覆盖不足，光照漂移 | 先补 hard negative，再看 illumination / robustness |
| 电池极片/涂布/焊缝 | 边缘亮带、拼接线、辊印、tile seam | 分辨率丢失或 tile overlap 不足 | 先保分辨率，再加 overlap 和 weighted blending |
| 少样本/紧急上线 | `normal` 太少、`normal` 里混脏样本、阈值直接照搬 | 数据本身不稳，不是参数没调够 | 先做干净基线，再看少量 hard negative 是否可控 |

这里有一个非常常见的误判：

- 看到误报多，就立即调 `threshold`。
- 但真实项目里，更常见的根因其实是 ROI、对齐、分组、分辨率、光照这几件事没先处理好。

如果你已经确定是后处理问题，再去看：

- `docs/FALSE_POSITIVE_DEBUGGING.md`

## 16. 看到这些现象时，先调哪里

### 现象一：边框、夹具、背景大面积发热

优先动作：

```bash
--roi-xyxy-norm 0.1 0.1 0.9 0.9
--defect-border-ignore-px 2
```

先别急着抬高阈值。  
如果可检区域本来就只占画面中间，最有效的办法通常不是“调更狠”，而是“别让背景进 defects”。

### 现象二：满屏碎点、小白点、盐胡椒噪声

优先动作：

```bash
--defect-map-smoothing median --defect-map-smoothing-ksize 3
--defect-hysteresis
--defect-min-area 16
```

如果你一上来就把 `min_area` 设得很大，虽然误报会少，但真实小缺陷也很容易一起被吃掉。  
先用 smoothing + hysteresis 压 speckle，通常比单纯抬面积阈值更稳。

### 现象三：tile seam 像一条条假缺陷

优先动作：

```bash
--tile-size 512 --tile-stride 384 --tile-map-reduce hann
```

排查顺序建议是：

1. 先确认有没有 overlap
2. 再确认 blending 不是 `max`
3. 最后再决定要不要改 tile 尺寸

### 现象四：一个真实缺陷被切成很多碎块

优先动作：

```bash
--defect-merge-nearby
--defect-merge-nearby-max-gap-px 1
--defect-close-ksize 3
```

这类问题常见于细长缺陷、纹理缺陷和低对比缺陷。  
如果业务更关心一个 defect 对应一个 region，先处理连通性，再谈漂亮的形状过滤。

### 现象五：长条结构总被过滤掉

优先动作：

- 先检查是不是把 `max_aspect_ratio` 设得过小
- 再检查是不是用针对圆斑噪点的规则去过滤线状缺陷

这类项目里，长宽比本来就可能很极端。  
不要把“去掉长细误报”的经验直接照搬到裂纹、焊缝、极片边缘缺陷上。

### 现象六：换一批光照或换一班次后误报猛增

优先动作：

- 先确认这批 `normal` 是否出现在训练集 / 校准集里
- 再确认有没有做 illumination 相关预处理
- 最后再决定 threshold 是否需要重校准

这类问题如果直接靠手工阈值顶过去，通常只能短期止血，不能稳定上线。

## 17. 上线交付最小清单

如果你是要把结果交给另一个团队、服务端容器或产线系统，最低限度建议交这些东西。

### 必交产物

- `infer_config.json`
- 如果要跨机器复制，优先交 `deploy_bundle`
- 一份可直接运行的推理命令
- 一小包样例输入图
- 对应的样例输出：`results.jsonl`、mask、overlay

### 必写清楚的说明

- 这个配置是针对哪个工位、哪个类目、哪个视角
- 输入图像的尺寸、颜色格式、路径约定
- 是否要求 ROI、tiling、illumination preprocessing
- defects 输出里，业务最关心的是 score、mask 还是 region
- 目前接受的误报预算和主要风险点

### 最容易漏掉的东西

- 只交训练配置，不交 `infer_config.json`
- 没有交样例输出，导致下游不知道怎么接 `results.jsonl`
- 没写清楚当前阈值和 pixel-threshold 是怎么来的
- 没有留下 hardest normal 作为回归验收集

### 一个更稳的交付包建议

```text
handoff/
  infer_config.json
  run_command.sh
  sample_inputs/
  sample_outputs/
    results.jsonl
    masks/
    overlays/
  README_handoff.md
```

如果你准备把项目从“能跑”交到“能维护”，这份最小交付包通常比多给几张指标截图更有价值。

## 18. 按行业写 manifest：字段怎么落，元数据怎么留

如果项目只是临时试验，目录结构能跑就行。  
但只要你开始遇到下面任意一个问题，就应该尽快切到 manifest：

- 多视角
- 多工位
- 多类目
- 不同照明条件
- 需要避免同件产品泄漏到 train/test
- 需要长期维护和复盘

### 先记住这几个最值钱的字段

| 字段 | 作用 | 哪些场景最重要 |
|------|------|----------------|
| `image_path` | 图像路径 | 全部 |
| `category` | 类目 / 产品类型 | 全部 |
| `split` | `train` / `val` / `test` | 全部 |
| `label` | `0/1` | 全部 |
| `mask_path` | 缺陷 mask 路径 | 需要 pixel 指标时 |
| `group_id` | 防止同件样本泄漏 | PCB、多视角、视频序列、连拍 |
| `meta.view_id` | 视角 / 相机位 | PCB、多相机工位 |
| `meta.condition` | 光照 / 工位 / 站点条件 | 纹理、反光、跨班次 |
| `meta.station_id` | 产线站点 | 多工位 |
| `meta.shift` | 班次 | 光照或工况差明显时 |
| `meta.lot_id` | 批次 | 材料差异大的项目 |
| `meta.template_id` | 模板 / 版式编号 | 标签、印刷、喷码 |

### 样例一：金属、玻璃、涂装表面

这类项目通常先保证 ROI 和 lot 维度可追溯：

```jsonl
{"image_path":"images/train/good_0001.png","category":"cover_glass","split":"train","label":0,"group_id":"lot_a","meta":{"station_id":"s1","shift":"day","lot_id":"lot_a"}}
{"image_path":"images/test/good_0101.png","category":"cover_glass","split":"test","label":0,"group_id":"lot_b","meta":{"station_id":"s1","shift":"night","lot_id":"lot_b"}}
{"image_path":"images/test/ng_0201.png","category":"cover_glass","split":"test","label":1,"mask_path":"masks/ng_0201.png","group_id":"lot_b","meta":{"station_id":"s1","shift":"night","lot_id":"lot_b"}}
```

建议：

- 如果反光、批次差异明显，`lot_id` 很有价值。
- 如果同一工件会连拍多张，`group_id` 最好绑定到同一件产品，而不是单张图。

### 样例二：标签、印刷、喷码

这类项目最容易缺的是模板和版式信息：

```jsonl
{"image_path":"images/train/normal_0001.png","category":"label_print","split":"train","label":0,"group_id":"roll_01","meta":{"template_id":"tpl_a","station_id":"printer_2","shift":"day"}}
{"image_path":"images/test/normal_0101.png","category":"label_print","split":"test","label":0,"group_id":"roll_02","meta":{"template_id":"tpl_a","station_id":"printer_2","shift":"night"}}
{"image_path":"images/test/ng_0201.png","category":"label_print","split":"test","label":1,"mask_path":"masks/ng_0201.png","group_id":"roll_02","meta":{"template_id":"tpl_a","station_id":"printer_2","shift":"night"}}
```

建议：

- `template_id` 不要省，不然你后面很难解释某个模板为什么误报突然升高。
- 如果会换打印头或换卷材，也建议继续扩展 `meta`，比如 `print_head_id`、`material_id`。

### 样例三：PCB、连接器、电子件多视角

这类项目最容易出错的是把同一件产品不同视角拆散，造成泄漏：

```jsonl
{"image_path":"images/item_1001_cam0.png","category":"pcb","split":"train","label":0,"group_id":"item_1001","meta":{"view_id":"cam0","condition":"station_a","station_id":"a"}}
{"image_path":"images/item_1001_cam1.png","category":"pcb","split":"train","label":0,"group_id":"item_1001","meta":{"view_id":"cam1","condition":"station_a","station_id":"a"}}
{"image_path":"images/item_2001_cam0.png","category":"pcb","split":"test","label":1,"mask_path":"masks/item_2001_cam0.png","group_id":"item_2001","meta":{"view_id":"cam0","condition":"station_b","station_id":"b"}}
{"image_path":"images/item_2001_cam1.png","category":"pcb","split":"test","label":1,"mask_path":"masks/item_2001_cam1.png","group_id":"item_2001","meta":{"view_id":"cam1","condition":"station_b","station_id":"b"}}
```

建议：

- 同一件产品的多视角必须共享 `group_id`。
- `view_id` 要稳定，不要今天写 `left_cam`、明天写 `cam_left`。
- 如果不同工位光照不同，用 `condition` 或 `station_id` 留痕，后面排查最省时间。

### 三条硬规则

1. 相对路径优先，方便迁移和交付。
2. 同一物理实体的多张图，优先共享 `group_id`。
3. 任何以后可能拿来切分析维度的信息，都尽量早放进 `meta`。

## 19. 可直接复制的验收记录模板

真正做项目时，很多问题不是“模型没调好”，而是没人把这次验收到底怎么验、验出了什么、剩什么风险写下来。  
下面这个模板可以直接丢到项目文档里。

```markdown
# 异常检测项目验收记录

## 1. 项目基本信息

- 项目名：
- 场景类型：
- 产品 / 类目：
- 工位 / 相机：
- 验收日期：
- 验收人：

## 2. 数据快照

- manifest 路径：
- train normal 数量：
- test normal 数量：
- test anomaly 数量：
- 是否有 mask：
- hard negative 集合说明：
- 是否做了 group-aware 拆分：

## 3. 本次采用方案

- run_dir：
- infer_config 路径：
- 模型：
- 是否启用 ROI：
- 是否启用 tiling：
- 是否启用 defects：
- 关键阈值 / pixel-threshold 来源：

## 4. 验收口径

- 允许的误报预算：
- 最低召回要求：
- 是否要求输出 mask：
- 是否要求输出 region：
- 是否要求最差视角单独达标：

## 5. 验收结果

- image-level 结果：
- pixel-level 结果：
- hard negative 表现：
- 最差视角 / 最差工位表现：
- overlays 抽检结论：

## 6. 未解决风险

- 当前最主要误报来源：
- 当前最主要漏检来源：
- 哪些条件下结果仍不稳定：

## 7. 交付清单

- infer_config.json：
- deploy_bundle：
- sample_inputs：
- sample_outputs：
- 运行命令：

## 8. 结论

- 是否允许试生产：
- 是否允许正式上线：
- 下一轮优先动作：
```

这份模板的重点不是“写完整”，而是把这次决策为什么成立留痕。  
后面如果效果回退，你能第一时间知道是数据变了、工况变了，还是配置本身就没站稳。

## 20. 首次进线前 30 分钟检查清单

如果模型已经在离线环境跑通，但你准备第一次接近真实产线或交给现场同学使用，建议至少过一遍下面这张表。

### 配置检查

- `infer_config.json` 路径是否正确
- checkpoint / artifact 路径是否能在当前机器解析
- 输入尺寸、颜色格式、位深是否和离线一致
- 如果是 12-bit / 16-bit 灰度图，是否明确了 `u16_max`

### 产物检查

- 是否能稳定产出 `results.jsonl`
- 如果业务要 defects，是否能稳定产出 mask 和 overlay
- 下游是否真的能读懂 JSONL 字段
- 样例输出是否已经给到现场

### 数据检查

- 现场输入是否混入了离线从未见过的新视角 / 新工位 / 新模板
- 是否有一小批 hardest normal 可以现场先跑
- 如果是多视角项目，当前视角标识是否仍然一致

### 风险检查

- 误报预算是否已经和业务约定
- threshold / pixel-threshold 是否有来源记录
- 如果误报突然升高，现场先看哪个目录下的 overlay，是否已经写清楚
- 如果需要回滚，回滚到哪个 `infer_config`，是否已经准备好

### 最小现场命令

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

第一次进线时，不要直接上全量目录。  
先拿一小包 hardest normal 和几张代表性 anomaly 做核对，确认输入格式、路径、产物、下游消费都没偏，再扩大规模。

## 21. 术语与约定

这一节按 reference 文档的方式把高频术语解释清楚。  
如果你是第一次接触 `pyimgano`，建议把这一节当作术语索引来查。

### 核心术语

| 术语 | 含义 | 什么时候最重要 |
|------|------|----------------|
| `normal` | 正常样本，通常用于训练、校准和误报控制 | 全部 |
| `anomaly` | 异常样本，通常用于测试和验收 | 全部 |
| `hard negative` | 看起来很像异常，但业务上应判正常的样本 | 误报控制、上线前验收 |
| `pixel_map` | 像素级异常热力图 | 要导出 mask / defects 时 |
| `defects` | 从 anomaly map 进一步导出的二值 mask 和 region 信息 | 工业质检交付 |
| `ROI` | 可检区域限制，只让局部区域参与缺陷导出或统计 | 背景复杂、夹具明显时 |
| `tiling` | 把大图切成多个 tile 再推理 | 2K/4K、大图小缺陷 |
| `overlay` | 原图 + 热力图 + mask 的调试可视化 | 排查误报、给业务解释 |
| `infer_config.json` | 部署式推理配置，包含模型和推理所需关键信息 | 交付、部署、复现 |
| `deploy_bundle` | 更适合跨机器交付的推理包，通常包含 `infer_config.json` 和依赖 artifact | 部署、容器化、跨团队交接 |
| `group_id` | 用来防止同一物理实体同时出现在 train 和 test | 多视角、连拍、视频片段 |
| `meta` | 用户自定义元数据，用于保留视角、工位、班次、模板等分析维度 | 真实项目长期维护 |

> 提示
>
> 如果你后面希望按“视角 / 工位 / 批次 / 模板”做误报复盘，最稳妥的办法不是事后补字段，而是第一次做 manifest 时就把这些信息写进 `meta`。

### 规则约定

下面这些约定在整份手册里保持一致：

- `normal` 默认指业务可放行样本，不等于“视觉上绝对完美”。
- `缺陷导出` 默认指 `--defects` 输出的 mask 和 region，而不是单纯的 image score。
- `上线` 默认指配置已经开始被现场或下游系统消费，而不是你本地把命令跑通。
- `首轮` 默认指不追求极致最优，而追求稳定、可解释、可交付的第一版结果。

### 命名建议

为了后续文档、数据和产物统一，推荐尽量保持这些命名习惯：

- `cfg_manifest.json` 表示 workbench 训练配置
- `infer_config.json` 表示部署式推理配置
- `sample_inputs/` 表示样例输入
- `sample_outputs/` 表示样例输出
- `view_id`、`station_id`、`lot_id`、`template_id` 尽量用稳定短字符串，不要一会中文一会英文

## 22. 常见问答

这一节按 FAQ 的写法整理最常见的问题，风格会更接近 GitHub / PyTorch 那种“问题 -> 结论 -> 简短解释”。

### 我应该用 `--model` 直接推理，还是优先用 `--infer-config`？

结论：

- 做临时试验、模板型快速验证时，可以直接用 `--model`
- 只要进入真实项目交付，优先用 `--infer-config`

原因：

- `--model` 更适合探索
- `--infer-config` 更适合复现、审计、交付和跨机器部署

### 我什么时候必须关心 `pixel_map`？

当你有下面任意一个需求时，就必须优先关心 `pixel_map`：

- 要输出 mask
- 要输出 region
- 要做缺陷定位
- 要看 overlay 排查误报

如果你只需要一个 image-level score，`pixel_map` 不是强制项。  
但在工业项目里，真正完全不看定位信息的情况并不多。

### 为什么离线评估还行，到了现场误报突然变多？

最常见的原因不是“模型坏了”，而是下面几类偏差：

- 现场输入尺寸、颜色格式、位深和离线不一致
- 新工位 / 新视角 / 新模板没有出现在训练或校准数据里
- ROI、tiling、illumination 这类推理约束没有同步到现场
- 阈值来源没有留痕，现场直接沿用了不适合当前条件的旧值

排查顺序建议：

1. 先看输入格式
2. 再看 `infer_config.json`
3. 再看 overlay
4. 最后才去怀疑阈值和模型

### `group_id` 和 `meta.group_id` 有什么区别？

从“防泄漏”的用途上看，它们都可以表达组信息。  
但从工程维护角度，优先推荐直接用顶层 `group_id`，因为更直观，也更不容易被遗漏。

只有当你历史数据格式已经把所有自定义字段都塞进 `meta` 时，才考虑继续沿用 `meta.group_id`。

### 我什么时候需要 `mask_path`？

当你要做下面这些事时，就需要尽量提供 `mask_path`：

- pixel AUROC
- pixel AP
- AUPRO
- mask 级验收

如果项目里完全没有 pixel 级 GT，也能先做第一轮落地。  
但这时你就要更依赖：

- hard negative 集合
- overlay 抽检
- 误报预算
- region 形态是否合理

### 我应该交整个 `runs/` 目录，还是只交 `deploy_bundle`？

结论：

- 做内部排查和完整复盘时，`runs/` 更完整
- 做交付和部署时，`deploy_bundle` 更稳

因为 `deploy_bundle` 更像一个可搬运、可消费的最小推理包，目标更明确，不容易把无关训练产物一起带过去。

### 什么时候应该重新校准 threshold / pixel-threshold？

如果出现下面任意一个信号，就应该重新审视阈值来源：

- 新班次、新工位、新光照进入系统
- `normal` 的分布明显变化
- 误报突然显著增加
- 现场输入格式和历史输入不再一致

但要注意：

- 重新校准阈值，不等于拿几张现场图手工试到“看着顺眼”
- 更稳的做法是保留一小批现场 `normal`，按同一规则重新校准并留痕

### 为什么我已经开了 `--defects`，但结果还是不好用？

通常是以下几类问题：

- 模型本身没有稳定的 `pixel_map`
- ROI 没设对
- 图太大但没有 tiling
- 后处理对真实缺陷不友好，比如线状缺陷被面积或形状过滤掉
- 只看数值，不看 overlays

`--defects` 只是把 map 变成 mask / regions，它不会自动修正前面的数据和推理问题。

## 23. 常用任务索引

这一节按“我现在要做什么”来组织命令，更接近文档站里的 quick reference。

### 校验部署配置

```bash
pyimgano-validate-infer-config /path/to/infer_config.json
```

适用场景：

- 推理前确认配置合法
- 跨机器复制后确认路径仍可解析
- 正式进线前做最后一次静态检查

### 从 workbench 训练并导出部署配置

```bash
pyimgano-train --config ./cfg_manifest.json --preflight
pyimgano-train --config ./cfg_manifest.json --export-infer-config
```

适用场景：

- 首轮建立稳定基线
- 需要产出可复现的 `infer_config.json`
- 后面还要走部署式推理

### 用导出的配置做部署式推理

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/results.jsonl
```

适用场景：

- 真实项目推理
- 交付给下游
- 验证离线和现场是否一致

### 导出缺陷 mask 和 overlay

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks \
  --save-overlays /tmp/overlays
```

适用场景：

- 误报排查
- 业务需要 defect mask / region
- 需要给现场或客户解释为什么判异常

### 直接试模板型基线

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/normal \
  --input /path/to/images \
  --defects-preset industrial-defects-fp40 \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks
```

适用场景：

- 标签、印刷、喷码
- 稳定站位
- 想先快速知道模板法能不能打

### 大图 tiling 推理

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --tile-size 512 \
  --tile-stride 384 \
  --tile-map-reduce hann \
  --save-jsonl /tmp/results.jsonl
```

适用场景：

- 2K / 4K 图像
- 小缺陷很容易被缩放吃掉
- tile seam 需要控制

### 现场最小 smoke run

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

适用场景：

- 第一次进线
- 新环境部署后验收
- 现场快速确认输入和产物链路

## 24. 进一步阅读

如果你希望继续按“文档站”的方式往下看，推荐顺序如下：

- 通用落地路径：`docs/ZH_ZERO_TO_ONE_GUIDE.md`
- 中文 FAQ / 排障：`docs/ZH_FAQ_TROUBLESHOOTING.md`
- 部署式推理：`docs/INDUSTRIAL_QUICKPATH.md`
- manifest 数据集：`docs/MANIFEST_DATASET.md`
- 误报排查：`docs/FALSE_POSITIVE_DEBUGGING.md`
- CLI 选项总表：`docs/CLI_REFERENCE.md`

如果你准备继续扩写这份文档，比较自然的下一步通常是：

- 再补一个“行业样例配置库”中文文档
- 再补一个“现场交付模板包”中文文档
- 再补一个“多角色协作与交接”中文文档
