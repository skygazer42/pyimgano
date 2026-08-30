# 论文实现校对审计（2026-08-30）

审计基线：`53eb969d25f4c9c9ec784765cabc11ccbe203e23`（`main`）

## 结论

本报告在审计基线上发现的**可由仓库内代码修复的问题已经全部核销**：公式、分数方向、查询批次依赖、OpenCLIP layout/tokenizer、训练选择协议、checkpoint 契约、第三方 notice、代理算法误标和安全扫描问题均已有实现或 fail-closed 修复，并增加了对应的不变量回归测试。

这仍不等于“所有论文指标已经复现”。需要官方权重、外部训练器、专有数据或完整 MVTec AD/VisA 重训才能回答的问题，现已作为 external backend、paper profile 或未认证数值边界显式登记，不再伪装成本地已验证能力。这里的“全部核销”指本报告 finding 均已获得代码修复、拒绝不安全/不完整输入，或准确的能力边界；不把无法在仓库内完成的外部实验冒充修复结果。

## 当前工作树修复状态

本节记录审计基线之后的最终核销；后文各 finding 和模块矩阵保留的是**基线证据与当时行号**，用于追溯问题来源，不能再当作当前实现状态。

| 范围 | 当前状态 |
|---|---|
| H-01 / M-10 OpenCLIP | 已按 attention capability 显式选择 BNC/LNB layout，未知实现 fail-closed；tokenizer 改为 model-specific；自定义 KNN backend 不再被忽略；WinCLIP/AdaCLIP/AA-CLIP/InCTRL/PromptAD 共用该判定并有双 layout 回归测试 |
| H-02/H-04/H-05/H-06、M-01/M-09 | 已修正 LID reciprocal MLE、ExtraTrees 训练分母、PCA 非退化默认、QMCD 双侧稳健 OOS 分数、LoOP zero-mean RMS、LODA 重复除数；均有数值不变量测试 |
| H-03 | LOCI/SOS/IMDD/LMDD 改为使用拟合训练状态的确定性 novelty extension；同一样本的分数不再依赖查询同伴、batch 大小或调用次数 |
| H-07/H-08/H-10 | AA-CLIP 恢复论文/作者的 image-pixel 融合；One-for-More 恢复 8 层 AP smoothing reducer；CFlow 使用训练期固定 likelihood normalizer，消除 query-batch 依赖 |
| H-09 | 10 个 PyOD 派生文件增加固定 commit 来源标记，`third_party/NOTICE.md` 保留 BSD-2-Clause 全文；审计工具增加不可由删 marker 绕过的派生文件清单 |
| M-04/M-05/M-11/M-12/M-17 | FiLo++ 拒绝零 key coverage；PatchCore-Inspection 校验 `input_shape`；AnomalyDINO schema v2 保存完整推理配置并披露 architecture-source 边界；anomalib model-specific aliases 验证模型身份且依赖限制为 `<3.0.0`；PatchCore 恢复实际 `n_neighbors`；EfficientAD map 恢复原图尺寸；PaDiM schema v4 封闭 extractor/preprocess/Gaussian contract；SPADE 不再恢复保存机器的 device |
| M-15/M-22 | PANDA 恢复 `decision_function -> score`、`predict -> label`；MemAE 对 uint8 `[0,255]` 与 float `[0,1]` 使用等价归一化 |
| M-02/M-03/M-18 | DRAEM 恢复论文的十种增强池与三种无放回抽样；RealNet 支持每 epoch 在线 SDAS sampler；ALAD 默认使用 validation holdout、patience=10、最佳 feature-score 权重恢复 |
| M-06/M-07/M-08/M-13/M-14/M-19--M-21 | offline-safe 默认与 paper profile 分层；下载入口增加显式 gate；SUOD、Bayes-PFL、WinCLIP、RegAD、GLAD、AST、DeepSVDD、FCDD 准确披露加速层、外部 backend、独立复原、权重和 benchmark 边界 |
| M-16 经典算法 | CBLOF 非法分割 fail-closed；Feature Bagging 恢复论文子空间与 cumulative/breadth-first 组合；HBOS 训练范围外保持异常性；HST 恢复工作空间、中点树、reference/latest 窗口与论文 mass score；INNE/SOD 边界公式修复；LSCP 在标准化空间构造局部域；RGraph 恢复稀疏自表示随机游走；ROD 默认枚举全部三维组合；RRCF 恢复 range-proportional cut 与 codisp。ODIN/SUOD/DBSCAN/SSIM 的库级扩展保留为明确 adaptation |
| 工程与安全追加项 | 全 registry 元数据合同已清零；pickle 加载改为显式 `trusted=True`；bundle watch 下载做 DNS/IP pinning、私网/重定向拒绝；OpenCLIP 默认禁止隐式下载；严格分数方向审计覆盖全部非深度 `core_*` |

保留的外部边界：本轮没有重训 MVTec AD/VisA 全基准；Bayes-PFL 等 external backend 的官方数值、RegAD 的十轮/oracle benchmark、GLAD 的官方 checkpoint 数值等不能由不含这些工件的仓库独立认证；AnomalyDINO schema-v2 虽保存 state/config/memory bank，恢复 backbone 仍需匹配的 DINOv2 architecture source。这些边界均已进入 metadata 或错误消息，不再属于静默缺陷。

当前自动门禁：

- metadata contract：279 个模型，required/recommended/invalid 均为 0；
- score direction：64 个非深度核心模型，warning 0；
- Semgrep `p/python + p/security-audit`：200 条规则、1083 个目标文件（含未跟踪文件），finding/error 均为 0；
- 可选后端：ONNXScript、diffusers、OpenCLIP、anomalib 均已安装并实际执行，原 6 个依赖缺失 skip 已消除；
- 全量测试：2887 passed、0 skipped、0 warnings、0 failed（1055.02 秒）。
- 依赖漏洞：安全刷新后的完整可选 profile 只剩 Lightning 2.6.5 的 `PYSEC-2026-3624` / `CVE-2026-58659`，上游尚无已发布修复版；例外、暴露面和缓解措施已写入 `SECURITY.md`，CI 对除此之外的任何 finding 阻断。

### 最终 finding 核销清单

| Finding | 核销结果 |
|---|---|
| H-01 | 修复 OpenCLIP BNC/LNB capability 判定；未知 layout fail-closed |
| H-02 | 修复 LID reciprocal MLE 与排序方向 |
| H-03 | LOCI/SOS/IMDD/LMDD 使用拟合状态，消除 query-context 与重复调用漂移 |
| H-04 | ExtraTrees density 固定使用训练样本分母 |
| H-05 | PCA 默认不再全成分精确重建，并纠正论文归属/适配声明 |
| H-06 | QMCD 使用双侧稳健 OOS 分数并纠正来源声明 |
| H-07 | AA-CLIP 恢复 image/pixel 融合 |
| H-08 | One-for-More 恢复八层 AP smoothing reducer |
| H-09 | PyOD 固定来源、BSD-2-Clause notice 与派生文件强制审计已补齐 |
| H-10 | CFlow 固定使用训练期 likelihood normalizer |
| C-01 | STFPM 明确 paper/source 分歧与 `pretrained_teacher=True` paper profile |
| C-02 | SimpleNet 明确 offline-safe 默认与 `pretrained=True` paper profile |
| C-03 | SPADE 保留论文平方距离，并明确与公开二次实现的差异 |
| M-01 | LoOP nPLOF 改为论文 zero-mean RMS |
| M-02 | DRAEM 恢复十种增强池、三种无放回抽样和旋转分布 |
| M-03 | RealNet 支持每 epoch 在线 sampler；官方 diffusion SDAS 保留为显式外部依赖 |
| M-04 | FiLo/FiLo++ checkpoint 零 key coverage fail-closed |
| M-05 | PatchCore-Inspection checkpoint 验证 input/preprocess 身份 |
| M-06 | SUOD 准确登记为 score ensemble，不再声称三层加速系统 |
| M-07 | 预训练下载增加显式 gate；offline-safe 与 paper profile 分层 |
| M-08 | Bayes-PFL 明确登记为用户提供 bridge 的 external backend facade |
| M-09 | LODA 去除重复投影数除法 |
| M-10 | OpenCLIP 使用 model-specific tokenizer，并实际使用注入 KNN backend |
| M-11 | AnomalyDINO schema-v2 保存全部推理配置；DINOv2 architecture-source 边界显式登记 |
| M-12 | anomalib model-specific aliases 验证 checkpoint 身份，并限制支持 major 版本 |
| M-13 | WinCLIP 改为“独立论文复原”声明并披露未验证参数/权重边界 |
| M-14 | RegAD/GLAD 明确单次适配、oracle benchmark 与 checkpoint 数值边界 |
| M-15 | PANDA 恢复 `decision_function=score`、`predict=label` 公共契约 |
| M-16 | 所有确定性经典算法偏差已修复；原本就不是论文 anomaly/OOS 定义的扩展改为 honest adaptation |
| M-17 | EfficientAD/PatchCore/PaDiM/SPADE 的 map 与 checkpoint 契约已封闭 |
| M-18 | ALAD 默认 validation early-stop、best-state restore 已实现 |
| M-19 | AST 明确随机冻结 offline 默认与 ImageNet paper profile |
| M-20 | DeepSVDD 明确 objective core 与图像 CNN/AE proxy 边界 |
| M-21 | FCDD 明确默认 backbone 与 pretrained/frozen paper profile |
| M-22 | MemAE 的 uint8 与 float 图像归一化语义统一 |

## 范围与口径

- 当前注册表共有 279 个 entry；以下候选、模块和引用覆盖数字均为审计基线统计。
- 论文审计候选共有 143 个 entry：115 个含 `paper`，7 个含 `related_paper`，另有 21 个 deep entry 没有论文引用但纳入了“是否误称论文实现”的反向检查。
- 上述 entry 落在 84 个 Python 模块；其中 70 个模块有直接或关联论文引用，30 个是经典/特征算法模块，40 个是 deep/VLM/外部适配模块。21 个无引用 deep entry 分布于 15 个模块，其中 14 个模块不与前述 70 个论文引用模块重叠；另 1 个是同时承载有引用/无引用 aliases 的 `anomalib_backend`。
- Registry 并不是全部来源：HST、ODIN、RGraph、ROD、RRCF、SOD、SOS 等 source/docstring 明确对应论文算法，却没有完整 `paper` metadata。因此经典算法分支又扩展检查了 64 个 classical `core_*` entry（并跟踪其 vision wrappers），而不是只检查前述 30 个 registry-paper modules。
- 基线的 115 个直接论文 entry 中只有 45 个带 `paper_url`，只有 8 个带 `official_code_url` 或 `official_repository`；这是当时的来源覆盖证据。
- 基线有 48 个论文 entry 缺少 `paper_fidelity`。当前 registry 的 metadata contract 审计已没有 required、recommended 或 invalid 项。
- 审计以“模块”为公式/实现单位；共享同一 constructor 的 registry aliases 合并判断。

状态含义：

- `对齐`：本次静态对照未发现核心公式/网络/推理路径偏差；不等于复现论文指标。
- `条件对齐`：核心路径一致，但依赖版本、checkpoint、预处理或用户注入组件决定最终行为。
- `明确适配`：有意简化或改造，并且 metadata/docstring 基本如实披露。
- `声明过强`：实现可能有用，但当前 paper-fidelity/implementation-status 容易被误读为论文复现。
- `缺陷`：存在可重复的公式、分数、状态或兼容性错误。
- `外部代理`：本地只负责调用外部仓库/checkpoint，不能由本仓库测试证明论文一致性。
- `未证实`：缺少可执行 checkpoint、官方基准配置或端到端证据；不把“没发现”写成“已通过”。

## 方法与证据规则

1. 从 registry AST/introspection 建立 entry → constructor → module → paper/fidelity 的清单。
2. 优先对照论文原文、作者官方仓库和固定 commit；博客、二次实现只用于发现线索，不作为最终定论。
3. 分开检查公式/网络结构、训练协议、预处理、推理聚合、分数方向、out-of-sample API、checkpoint schema 和 metadata 声明。
4. 对高风险疑点运行最小数值复现；没有权重或数据时明确写成“静态证据”或“未证实”。
5. 现有测试通过只记为工程契约证据，不把 smoke test 当成 paper-fidelity 证据。

## 高优先级发现

### H-01 — OpenCLIP 支持范围内存在静默 token-layout 错误

位置：

- `pyproject.toml:97` 声明 `open_clip_torch>=2.16.0`，没有上限或兼容版本分支。
- `pyimgano/models/openclip_backend.py:141-159` 先直接按 `(B,N,C)` 调 transformer，只有抛异常才尝试 `(N,B,C)`。
- `pyimgano/models/winclip.py:280-308` 固定走另一种 layout。
- `pyimgano/models/adaclip.py:80-93`（AA-CLIP 复用）、`pyimgano/models/inctrl.py:353-370` 依赖 `attn.batch_first`，缺少该属性时默认按旧 layout。
- `pyimgano/models/promptad.py:478-481,515-520` 假定旧式 `nn.MultiheadAttention` / layout。

OpenCLIP v2.16.0 的 ViT 在 transformer 前后显式 BNC→LNB→BNC；当前 OpenCLIP 已使用 batch-first 自定义 Attention。两种实现都能接受三维 tensor，错误 layout 往往不抛异常，所以异常驱动的探测会静默产生数值错误。旧版下通用 patch backend 错；新版下 WinCLIP、AdaCLIP、AA-CLIP、InCTRL/PromptAD 的部分路径错或不兼容。

固定 `torch.manual_seed(0)`、`dropout=0`、eval mode 的最小复现使用 `TransformerEncoderLayer(batch_first=False)`：local helper 和显式正确 permute 都成功并返回 `(1,5,4)`，但 `max_abs_diff=0.3811771869659424`。当 batch=1 时，错误路径会把 batch 维当注意力序列维，patch 之间没有按预期交互。完整 fixture 在 `tools/repro_paper_audit.py`。

主源：[OpenCLIP v2.16-era implementation](https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/open_clip/transformer.py#L458-L501)，[current batch-first implementation](https://github.com/mlfoundations/open_clip/blob/4a4e060bb2a5afbb9c59b882f09edb78f65a3b38/src/open_clip/transformer.py#L822-L837)。

影响：`vision_openclip_patch_map`、`vision_openclip_promptscore`、`vision_openclip_patchknn`，以及依赖真实 OpenCLIP block 的 WinCLIP、AdaCLIP、AA-CLIP、InCTRL、PromptAD。基线 optional test 只验 shape/finite，基线验证环境又因未安装 OpenCLIP 而 skip；fake attention 都显式暴露旧式 `batch_first`，没有覆盖真实版本矩阵。

### H-02 — LID 实现漏掉倒数，排序方向也随之反转

`pyimgano/models/lid.py:28-34` 返回：

```text
-mean(log(r_i / r_k))
```

LID/Hill MLE 是：

```text
-(mean(log(r_i / r_k)))^-1
```

对距离 `[1,2,4]`，当前结果是 `0.6931471805599453`，MLE 应为 `1.4426950408889634`。这不是常数缩放；它是正数倒数，因此在 metadata/docstring 的“higher LID = more anomalous”口径下会反转排序。

主源：[Amsaleg et al., *Estimating Local Intrinsic Dimensionality*, §4.1](https://mistis.inrialpes.fr/~girard/Fichiers/p29-amsaleg.pdf)。

### H-03 — 多个 dataset-level 算法被包装成错误的 inductive `decision_function`

- LOCI：`pyimgano/models/loci.py:91-101`
- SOS：`pyimgano/models/sos.py:129-158`
- IMDD：`pyimgano/models/imdd.py:56-64`
- LMDD：`pyimgano/models/lmdd.py:28-67` 复用同一 IMDD core

这些 `decision_function(X)` 不使用 `fit()` 保存的训练样本，而是在查询 batch 内重新计算 dataset-level 分数。结果既不是基于训练集的 out-of-sample scoring，也不是稳定的逐样本函数。

相同离群取值 7 的复现结果如下；LOCI/SOS 使用二维 `[7,7]`，IMDD/LMDD 使用六维全 7。IMDD 的 singleton 与 context 分别由相同 seed、相同训练数据的新实例计算，以隔离查询 batch 组成；重复调用的不确定性另行测试。

| 模型 | 单独查询 | 与 39 个其他查询一起 |
|---|---:|---:|
| LOCI | 0.0 | 3.162277660168378 |
| SOS | 0.0 | 0.9999999999843281 |
| IMDD / LMDD core | 0.0 | 130.89550337020918 |

这类算法原论文可能本来就是 transductive/dataset-level；问题在于本地 API 把它们暴露成普通 fitted detector，却没有拒绝 OOS、说明 transductive 语义或实现稳定的 novelty extension。

IMDD/LMDD 还存在第二个状态问题：`pyimgano/models/imdd.py:77-81` 每次 `decision_function` 都从 estimator RNG 再抽一个 seed。固定 `random_state=0,n_iter=10`，对同一 20×6 query batch 连续调用两次，最大分数差为 `5.9124`。因此输出不仅依赖查询同伴，还依赖调用次数。

### H-04 — ExtraTrees density 把查询 batch 大小当成训练集大小

docstring 在 `pyimgano/models/extra_trees_density.py:7-12` 定义 `leaf_count / n_train`，但 `_score_from_leaf()` 在 `:94-112` 用当前查询的 `leaf.shape[0]` 作分母。固定脚本中同一个样本重复 10 次，首项分数从 `-2.2552785108967677` 变成 `0.047306582097277845`，增加 `2.3025850929940455`，即 `ln(10)`（浮点舍入内）。

这是原生 baseline 的直接实现错误，虽然它没有 paper claim，也会污染依赖它的 vision wrappers。

### H-05 — PCA 默认配置退化，且不是所引用 ICDM 2003 分类器

`pyimgano/models/pca.py:32-42,68-77,91-105` 默认 `n_components=None`、`n_selected_components=None`，随后用所有保留成分 inverse-transform。典型 `n>d` 数据因此被精确重建：40×5、seed 0 固定数据的训练分数最大仅 `6.7053176943786e-30`；选择一个成分后最大值为 `11.16857916605836`。

此外，注册表 `:109-117,152-160` 引用的 Shyu et al. ICDM 2003 方法是 robust principal-component classifier，分别利用解释 50% 变异的 major components 和 eigenvalue < 0.20 的 minor components计算距离，并不是这里的普通 PCA reconstruction error。

主源：[Shyu et al., *A Novel Anomaly Detection Scheme Based on Principal Component Classifier*](https://lweb.umkc.edu/chen/PDF/ICDM03_WS.pdf)。因此这是“无效默认值 + paper attribution 过强”两个独立问题。

### H-06 — QMCD 的论文归属和 OOS 分数方向均不成立

`pyimgano/models/qmcd.py:8-11,100-108` 把实现归给 “Fang, Hickernell, Winker, 2001 / Wrap-around L2-discrepancy of lattice rules”。可核实的 2001 主论文实际作者是 Fang 与 Ma，题为 *Wrap-Around L2-Discrepancy of Random Sampling, Latin Hypercube and Uniform Designs*，研究均匀试验设计，不是 anomaly detector。DOI：[10.1006/jcom.2001.0589](https://doi.org/10.1006/jcom.2001.0589)。

本地实现实际与 PyOD 的 QMCD detector 同源；仓库历史也明确写有 `feat: port QMCD/... off PyOD`。`pyimgano/models/qmcd.py:77-97` 用训练分数的 skew/kurtosis 决定一次全局 flip，但 MinMaxScaler 对远离训练范围的查询会外推。固定复现中中心点为 `-5.48245212226246`，两个极远点为 `-2972.644527817618`、`-2837.377852373511`，违反全库“higher = more anomalous”契约。

上游实现：[PyOD QMCD fixed source](https://github.com/yzhao062/pyod/blob/34f7996effac700a5166d882d5e94c6e6078fae3/pyod/models/qmcd.py)。当前 metadata 应标成 PyOD-derived / discrepancy-inspired，而不是把均匀设计论文写成直接 anomaly implementation。

### C-01 — STFPM 论文与发布源码互相冲突；本地 map 路径更接近论文

`pyimgano/models/stfpm.py:275-297` 把每一层 distance map 插值到最终 output size，再逐像素相乘。作者发布代码 commit `2598a5e35fd0` 的 `main.py:151-172` 则先统一到 64×64、相乘，再在评估阶段 `:131-137` resize 到 256×256。

双线性插值与逐点乘法不交换，因此两条路径确实会给出不同定位图和 max image score；但 BMVC 2021 论文 §3.3 Eq.(4) 与 §4.2 的文字描述是把各层 map 上采样到输入分辨率后相乘，和本地顺序一致。checkpoint selection 也有类似冲突：论文写选择验证 Eq.(1) error 最低者，发布源码实际使用验证 anomaly-map mean。另一个独立条件是 `pyimgano/models/stfpm.py:149` 默认 `pretrained_teacher=False`，而论文/作者源码使用 ImageNet-pretrained ResNet18；只有显式设为 `pretrained_teacher=True` 时，teacher 初始化才符合论文配置。

主源：[BMVC 2021 paper](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1273.pdf)，[STFPM official repository](https://github.com/gdwang08/STFPM/blob/2598a5e35fd02f2f9dcfd0f3e8249adc22320e59/main.py#L151-L172)。结论应是“`pretrained_teacher=True` 时 paper-aligned/source-divergent；默认 constructor 是 offline-safe deviation”：不能把本地 map 顺序判为错误，也不能无条件声称默认路径和论文、发布源码都 core/bitwise 对齐。

### C-02 — SimpleNet 的显式论文配置与作者发布代码不同，但本地默认也有离线偏差

SimpleNet 论文 p.5 §4.3 明确给出 ImageNet-pretrained backbone、batch=4、resize/crop=256/224、bias-free FC adapter、两组 Adam learning rate 1e-4/2e-4、weight decay 1e-5。本地 `pyimgano/models/simplenet.py:27-37,93-110,326-335` 在显式设置 `pretrained=True` 时符合这些值；但 constructor 默认 `pretrained=False`，因此默认路径不是论文训练配置。作者仓库 commit `351a2b8` 又使用 batch=8、329→288、带 bias 的 Linear 和 AdamW 默认 weight decay。

主源：[CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html)，[official run config](https://github.com/DonaldRR/SimpleNet/blob/351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b/run.sh#L12-L30)，[official adapter implementation](https://github.com/DonaldRR/SimpleNet/blob/351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b/simplenet.py#L59-L81)。因此结论应写成“`pretrained=True` 时 paper-aligned/source-divergent；默认 constructor 是 offline-safe deviation”，而不是无条件 `core-aligned` 或官方代码数值复现。

### H-07 — AA-CLIP 图像级 score 没有复现官方融合

`pyimgano/models/aaclip.py:385-407,576-579,736-747` 返回 detection-only image score。AA-CLIP 官方推理先分别 min-max；Industrial score 是 `0.5 * max(pixel map) + 0.5 * image prediction`，Medical score 只用 pixel max。固定 detection logits、只改变 patch map 的 max 时，官方 score 改变，本地 `decision_function` 完全不变。

主源：[AA-CLIP official `forward_utils.py`](https://github.com/Mwxinnn/AA-CLIP/blob/53db195f230442aa118c246876c94ba1c76139cc/forward_utils.py#L241-L254)，[CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html)。registry 的 `paper-adaptation` 可以容纳部署改造，但 `native-paper-inference-openclip-adaptation` 应明确披露最终 score 已改变。

### H-08 — One-for-More 图像级 score 遗漏作者的 AP smoothing encoder

`pyimgano/models/oneformore.py:300-305` 对 sigma=5 平滑后的 anomaly map 直接取 raw max。作者评测把 map 交给 `apsp`；其中 `EvalImageAP` 连续 8 次执行 `avg_pool2d(kernel=8, stride=1)` 后才取 max。

最小复现把两个合成 map 直接输入 image-score aggregation stage：一个 256×256 map 只有单像素为 1，另一个有 64×64 方块为 1；它们不是声称由上游 sigma=5 路径实际生成，只用于隔离 score reducer。本地 raw max 都为 `1.0`，作者 encoder 分别得到 `0.0036432659` 与 `1.0`，因此两张图在本地完全同分、在发布评测 reducer 中明显不同。现有 `tests/test_oneformore.py:41-120` 使用常量 map 或让 fake backend 自己返回 raw max，恰好不能发现这个差异。

主源：[official evaluation selects `apsp`](https://github.com/FuNz-0/One-for-More/blob/f4eb78841dbfa5612e008570b690072b19a3d9b3/scripts/test_mvtec.py#L105-L112)，[official `EvalImageAP`](https://github.com/FuNz-0/One-for-More/blob/f4eb78841dbfa5612e008570b690072b19a3d9b3/utils/eval_helper.py#L210-L220)。定位 map、预处理和 reconstruction-vs-samples 路径本身未发现同等级偏差；结论只针对 image score。

### H-09 — PyOD 派生代码缺少第三方 notice/许可证保留

仓库历史明确包含：

- `642f062 feat: port QMCD/LMDD/ABOD/COF/LOCI off PyOD`
- `4b2636d feat: port HBOS/MCD/OCSVM/KPCA/INNE off PyOD`

当前 `third_party/NOTICE.md:1-15` 仍只是空模板；代码搜索没有 Yue Zhao、QMCD 作者 D Kulik 或 BSD-2-Clause notice。以 QMCD 为例，本地核心函数/fit/decision flow 与上游高度一致，但删掉了上游文件头的 author/license。PyOD 是 BSD-2-Clause，其 source redistribution 条款要求保留 copyright、conditions 和 disclaimer。

主源：[PyOD license](https://github.com/yzhao062/pyod/blob/34f7996effac700a5166d882d5e94c6e6078fae3/LICENSE)，[PyOD QMCD source](https://github.com/yzhao062/pyod/blob/34f7996effac700a5166d882d5e94c6e6078fae3/pyod/models/qmcd.py)。这是来源/合规审计发现，不是法律意见；但 `tools/audit_third_party_notices.py` 只检查显式 `UPSTREAM:` marker，因此“脚本通过”不能证明没有派生代码。

### C-03 — SPADE 本地平方距离符合论文，但与公开二次实现不同

Registry 在 `pyimgano/models/spade.py:78-102` 标为 `core-aligned`。本地：

- `pyimgano/models/spade.py:289-323` 用平方残差和作为 global KNN distance。
- `pyimgano/models/spade.py:401-410` 的 `cKDTree.query` 已返回 Euclidean distance，随后再次 `np.square`。

ArXiv 官方 LaTeX source 的 Eq.(2) 明确写成 `\|f-f_y\|^2`，Eq.(3) 写成 `\|f-F(y,p)\|^2`；因此本地平方距离正好符合论文。最小例子 gallery `[0],[4]`、query `[1]`、K=2，论文与本地的 global score 都是 `(1^2+3^2)/2=5`。

主源：[SPADE paper/source package](https://arxiv.org/abs/2005.02357)。作者没有发布官方仓库；[固定 commit 的公开二次实现](https://github.com/Byungjae89/SPADE-pytorch/blob/077c67be21d68a38b4442db7311c87e708728286/src/main.py#L110-L149) 使用非平方距离，只能证明该二次实现与论文分歧，不能据此给本地实现判错。现有 `tests/test_spade_algorithm.py:64-79,112-133` 对平方距离的断言与论文一致。

### H-10 — CFlow 的任意调用 API 复用了“完整评测集全局归一化”

`pyimgano/models/cflow.py:591-604` 对每个尺度在本次传入的整组图上取 `log_probability.amax()`，最后又对整组取 `probability_sum.amax()`。因此同一张图单独评分与和另一张极值更大的图一起评分时，map/score 会改变。`fit()` 在 `:533-536` 又用整套训练集的 normalization universe 校准 threshold，和单图 inference 的数值不可比。

作者发布代码也做全局 normalization，但只在一次固定、完整 evaluation set 上使用；直接移植到任意 `decision_function(batch)` 后语义不成立。主源：[CFlow official evaluation](https://github.com/gudovskiy/cflow-ad/blob/b2ebf9e673a0aa46992a3b18367ec066a57bba89/train.py#L308-L329)。这是 paper-to-library API adaptation 缺陷，应持久化 fit/validation normalizer 或使用逐图/固定校准，并在修复前把 `core-aligned` 降为 API-adapted。

## 中优先级发现

### M-01 — LoOP 的 nPLOF 使用 centered standard deviation

`pyimgano/models/loop.py:88-94` 使用 `np.std(plof)`。LoOP 定义里的 “standard deviation assuming a mean of 0” 对应 `sqrt(mean(PLOF^2))`，不是减去样本均值后的 centered std。当 PLOF 均值不恰好为零时，概率标定发生变化。主源：[LoOP paper DOI](https://doi.org/10.1145/1645953.1646195)。

### M-02 — DRAEM synthetic anomaly 分布明显简化

`pyimgano/models/draem.py:305-337` 只做 flip、channel permutation、gamma、brightness 和 90° Perlin rotation。官方 commit `2dbf67397ab5` 的 `data_loader.py:80-105,107-153` 每次从 10 种 augmentation 随机取 3 种，包含 sharpen、hue/saturation、solarize、posterize、invert、autocontrast、equalize、任意 affine/rotation 等。

主源：[DRAEM official data loader](https://github.com/VitjanZ/DRAEM/blob/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/data_loader.py#L80-L153)。`paper-adaptation` 是合理大类，但 `paper-network-and-schedule-aligned` 没有充分表达训练分布差异，不能期待复现论文指标。

### M-03 — RealNet 是 static preblended-pair approximation

`pyimgano/models/realnet.py:456-458,656-687` 要求用户提供已混合 normal/synthetic pair，并在固定 TensorDataset 上重复训练。官方 dataset 每次 `__getitem__` 随机选择 normal/SDAS、异常来源、Perlin mask 和透明度；官方 YAML 还规定 normal/SDAS 比例及透明度范围。

主源：[RealNet official dataset](https://github.com/cnulab/RealNet/blob/09e60a2ec50aa11560382c5961f1711088ed713a/datasets/realnet_dataset.py#L199-L254)，[official MVTec config](https://github.com/cnulab/RealNet/blob/09e60a2ec50aa11560382c5961f1711088ed713a/experiments/MVTec-AD/realnet.yaml#L14-L25)。registry 现有 `external-sdas` 说明方向正确，但应再明确“static preblended pairs”。

### M-04 — FiLo checkpoint 可以静默加载完全不匹配的权重

`pyimgano/models/filopp.py:256-259` 只检查 `checkpoint['filo']` 非空，`:285-286` 用 `strict=False` 且不检查 missing/unexpected keys。`{'filo': {'totally_wrong': tensor(...)}}` 可以越过 schema 检查，使所有 FiLo 参数保持构造时初始值而没有加载目标权重。官方本身也使用 `strict=False`，但本地作为 checkpoint adapter 应封闭工件契约。

主源：[FiLo official load path](https://github.com/CASIA-IVA-Lab/FiLo/blob/36ff29ca09ba8ba3af24d7654582aea856031400/test.py#L529-L536)。registry 对“legacy key 实际运行 FiLo 而非 FiLo++”的 compatibility note 是诚实的。

### M-05 — PatchCore-Inspection checkpoint 没有封闭预处理/版本契约

`pyimgano/models/patchcore_inspection_backend.py:16-57,60-108,136-169` 只检查工件文件名，用户可以独立指定 `imagesize`；官方 `.pkl` 保存 `input_shape`。例如用 224 工件初始化 `imagesize=320`，wrapper transform 与保存模型的输入契约不一致，却没有拒绝或采用 artifact 值。

主源：[official save/load implementation](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/patchcore.py#L234-L273)。该 `.pkl` 同时是可执行 pickle，只应接受可信来源。

### M-06 — `core_suod` 是 score ensemble，不是 SUOD acceleration system

`pyimgano/models/suod.py` 自己说明这是 simplified native ensemble，没有 random projection、pseudo-supervised approximation 或 load-balanced parallel scheduling；registry 的 `core_suod`、`core_suod_spec` 两个 entry 却直接绑定 SUOD 2021 论文，两个 vision wrappers 虽无 paper 字段也沿用 SUOD 命名。

论文的核心贡献就是三层 acceleration framework。主源：[MLSys 2021 paper](https://proceedings.mlsys.org/paper_files/paper/2021/file/37385144cac01dff38247ab11c119e3c-Paper.pdf)，[official repository](https://github.com/yzhao062/SUOD)。应改成 `inspired`/`related_paper`，并把本地算法命名为 heterogeneous score ensemble。

### M-07 — Offline-safe defaults 与 paper defaults 没有分层表达

`docs/PAPER_TO_MODULE_MAP_V4.md:10` 要求避免 implicit weight downloads；CLI 默认也显式设为 offline-safe。但模型层同时存在两个相反问题：AA-CLIP、AdaCLIP、PromptAD、WinCLIP 默认向 OpenCLIP 传非空 pretrained id，PANDA、RealNet、RegAD 默认 `pretrained=True`，没有统一的 `allow_download` / `local_files_only` gate；另一方面 PatchCore、PaDiM、SPADE、SoftPatch、CFlow、STFPM、SimpleNet、AST、FCDD 为了离线安全默认关闭论文要求的 ImageNet-pretrained encoder/teacher，却仍有多处 `core-aligned` 或 `paper-...-aligned` 声明。AST/FCDD 的额外冻结语义见 M-19/M-21。

这不代表所有 CLI 路径都会下载，也不代表关闭预训练后的网络结构没有价值；问题是“离线可构造默认值”和“论文复现 profile”没有成为两个明确配置。现有 CLI tests 只证明顶层 CLI 的 `--pretrained` 默认是 false，并未覆盖这些模型 constructor 的真实 backend 初始化或 paper profile。

### M-08 — Bayes-PFL 是用户自带 backend facade，不是官方 checkpoint adapter

`pyimgano/models/bayesianpf.py:53-64` 的说明和 registry backend 名称容易让人理解为官方 checkpoint-backed adapter；但 `:75-104` 在只传 `checkpoint_path`、没有自定义 backend 时立即报错，`:106-120,184-209` 也不实现官方预处理、prompt、score 或 checkpoint schema，只转调任意 callable。

官方 `test.py` 支持三种 CLIP backbone，严格读取 `checkpoint['MyModel']`；其发布脚本默认/公开权重配置是 OpenAI ViT-L/14@336。最小复现是 `VisionBayesianPF(checkpoint_path='train_visa.pth')` 仍然得到 backend-required RuntimeError。`external-backend` fidelity 可保留，但 description、weights source 和文档应写明“用户必须自行实现桥接器；本地不能直接加载官方工件”。主源：[Bayes-PFL fixed `test.py`](https://github.com/xiaozhen228/Bayes-PFL/blob/8f155a07e734913e021c33c469f16a1f75c60e5d/test.py#L83-L114)。

### M-09 — LODA score 被多除了一次投影数

`pyimgano/models/loda.py:167` 已把每个 projection 的 weight 设为 `1 / n_random_cuts`，`:209,231` 用该权重累加 `-log p`，但 `:235,264` 返回时又除一次 `n_random_cuts`。论文 Eq.(1)/Algorithm 2 是 `-(1/k) * sum(log p_i)`；本地实际为 `-(1/k^2) * sum(log p_i)`。

主源：[Pevný, *LODA: Lightweight on-line detector of anomalies*](https://link.springer.com/content/pdf/10.1007/s10994-015-5521-0.pdf)。这是确定的正比例尺度错误，不改变单模型内 rank/quantile threshold，但会改变分数解释、跨模型 ensemble 和任何依赖绝对尺度的校准。与 PyOD 的固定数据 rank 差异主要来自 RandomState/Generator 和 histogram edge convention，不能把该 rho 差本身当作另一项本地排序错误。

### M-10 — 通用 OpenCLIP 还有 tokenizer 与公开 KNN 参数契约问题

- `pyimgano/models/openclip_backend.py:341-360` 使用全局 `open_clip.tokenize`；当前 OpenCLIP 对 HF/SigLIP/TikToken 等模型要求按 model name 获取 tokenizer。默认 ViT-B/32 不受该项影响，但公开的可替换 model API 并不成立。
- `pyimgano/models/openclip_backend.py:689-713` 接受并保存 `knn_index`，后续却不使用该对象而始终自建索引。调用者注入的 backend 会被静默忽略。

主源：[OpenCLIP fixed `get_tokenizer`](https://github.com/mlfoundations/open_clip/blob/4a4e060bb2a5afbb9c59b882f09edb78f65a3b38/src/open_clip/factory.py#L833-L989)。

### M-11 — AnomalyDINO 保存工件不包含决定输出的配置

`pyimgano/models/anomalydino.py:264-310` 保存 payload 时遗漏 `aggregation_method` / `aggregation_topk`、masking、Gaussian sigma 与 KNN backend 等直接影响加载后推理的配置；加载后沿用接收对象的 constructor 参数，同一个 checkpoint 可以因此产生不同 map/score。reference rotations 等已经烘进 memory bank 的训练参数不会在 load 后直接改分，但其 provenance 同样没有记录。`:106-132,565-604` 还会在恢复 embedding state 前先通过 torch.hub 构建 backbone，因此工件不一定是自包含离线包。

其 448/DINOv2-S14、rotation/PCA foreground mask、cosine 1-NN、top-1% 和 sigma=4 默认方法路径与作者实现对齐。问题是 artifact reproducibility，不是这些核心公式。主源：[AnomalyDINO paper](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)，[official repository](https://github.com/dammsi/AnomalyDINO)。

### M-12 — Anomalib aliases 不验证 checkpoint 的实际模型身份

`pyimgano/models/anomalib_backend.py:92-198,201-534` 的 22 个 registry names 最终都进入同一个 `TorchInferencer` wrapper；alias 只提供发现标签，不验证 checkpoint 是 PatchCore、PaDiM、DRAEM 等哪一种模型。依赖又是 `anomalib>=0.10.0`、无 major 上限，因此 checkpoint/API 兼容性随环境漂移。

`external-backend` 分类是诚实的，`docs/ANOMALIB_CHECKPOINTS.md` 对 alias/pickle 风险也有说明；问题是 alias 不能当成已验证的论文模型身份。主源：[anomalib official repository](https://github.com/open-edge-platform/anomalib)。

### M-13 — WinCLIP 是按论文独立复原，不是可由官方代码验证的 native implementation

本次没有找到作者发布的官方代码或 checkpoint。本地完整 prompt、240px tiling 与 harmonic aggregation 可以在论文/补充材料中找到依据，但 `pyimgano/models/winclip.py:24-28,344-370` 的 `native-paper-method` 仍过于确定，`PAPER_TEMPERATURE=0.07` 也没有在主源中核实。再叠加 H-01 的 OpenCLIP 版本问题，只能标为“独立复原、条件未验证”。主源：[CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html)。

### M-14 — RegAD 与 GLAD 的核心路径可对照，但完整论文协议未复现/未验证

- RegAD 的 STN、SimSiam、Gaussian map 核心对齐；本地是单次 support 适配，没有复现作者十轮运行和 test-AUC oracle model selection。该 oracle 本身不适合生产，但 benchmark 数字不可直接对比。主源：[RegAD paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840300.pdf)，[official repository](https://github.com/MediaBrain-SJTU/RegAD)。
- GLAD 的 preset、ADS、SAFF 与输出语义能映射到作者路径，但本地用当前 diffusers DDIM 重建作者定制 pipeline；没有真实 checkpoint 的端到端数值对照，所以只标“方法适配、数值未证实”。主源：[GLAD paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08940.pdf)，[official repository](https://github.com/hyao1/GLAD)。

文档另有一个低风险事实错误：`docs/site/models/vlm.md:26` 把 OpenCLIP PatchKNN 的 registry key 写成了 `vision_openclip_patch_map`。

### M-15 — PANDA 的算法路径对齐，但 `predict()` 违反本库标签契约

PANDA-Early 的 ResNet152、2300 fixed steps、2-NN squared-L2 等核心路径与论文/作者实现对齐；`paper_fidelity=core-aligned` 有依据。但 `pyimgano/models/panda.py:294-324` 的 `predict()` 直接返回连续 anomaly score，`decision_function()` 又反过来调用 `predict()`。这与本库其他 detector 的 `predict -> {0,1}`、`decision_function -> score` 约定相反，也使 `_process_decision_scores()` 生成的 threshold 没有被 `predict()` 使用。现有 `tests/test_panda_paper_alignment.py:9-26` 只检查默认值和私有 scoring helper，没有检查公开 predict contract。

主源：[PANDA paper](https://openaccess.thecvf.com/content/CVPR2021/html/Reiss_PANDA_Adapting_Pretrained_Features_for_Anomaly_Detection_and_Segmentation_CVPR_2021_paper.html)，[official repository](https://github.com/talreiss/PANDA)。这是本地 API 错误，不是论文公式错误。

### M-16 — 其余经典论文算法的已证实偏差

| 模块 | 结论与本地证据 | 主源 |
|---|---|---|
| CBLOF | `cblof.py:209-217` 把 Pattern Recognition Letters 论文写成 “SDM 2003”；`:318-353` 在 alpha/beta 无法形成合法簇分割时静默选第一簇，而不是拒绝无效配置 | [原论文 DOI](https://doi.org/10.1016/S0167-8655%2803%2900003-5) |
| Feature Bagging | `feature_bagging.py:197-229` 默认子空间可取完整 `d`，论文上界为 `d-1`；`:255-264` 用 raw mean/max，未实现论文 cumulative breadth-first combination | [KDD 2005 DOI](https://doi.org/10.1145/1081870.1081891) |
| HBOS | `hbos.py:102-109` 把训练范围外值 clip 到边缘 bin；训练边缘高密度时，任意远的 OOD 点会继承“正常”低分。复现中 score(1)=score(100)=`0.106248` | [HBOS paper](https://www.dfki.de/fileadmin/user_upload/import/6431_HBOS-KI-2012.pdf) |
| HST | `hst.py:50-57,127-176` 使用训练 min/max、随机区间切点、仅叶 inverse-mass；原 HST 预构造工作空间、中点切分，使用 reference/current window mass 和 `mass*2^depth`。这是 unmarked proxy | [IJCAI 2011 paper](https://www.ijcai.org/Proceedings/11/Papers/254.pdf) |
| INNE | `inne.py:72-95,122-133` 接受 `max_samples=1`，随后出现 `inf/inf`，训练分数全 NaN；普通默认数据与上游 rank parity 1 不能覆盖该边界 | [ICDMW 2014 DOI](https://doi.org/10.1109/ICDMW.2014.70) |
| LSCP | `lscp.py:204-249` 在未标准化的原特征上建 local region，作者实现先标准化；source comment 称 inspired，但 registry 没有 fidelity | [SDM paper DOI](https://doi.org/10.1137/1.9781611975673.66)，[official repository](https://github.com/yzhao062/LSCP) |
| ODIN | `odin.py:73-88` 的训练分是自身 kNN 图 indegree；`:105-111` 对新样本改成“其邻居的平均 indegree”，不是原论文的 dataset-level ODIN 定义 | [ODIN paper DOI](https://doi.org/10.1109/ICPR.2004.1334558) |
| RGraph | `rgraph.py:1-16` 自己说明是 kNN proxy；`:50-54,260-302` 接受大量论文参数却不使用，也没有 self-representation optimization | [CVPR 2017 paper](https://openaccess.thecvf.com/content_cvpr_2017/html/You_Provable_Self-Representation_Based_CVPR_2017_paper.html) |
| ROD | `rod.py:178-213,275-321` 高维时最多随机采样 256 个三维子空间；例如 `d=20` 时论文式 1140 个三元组只取 256，是明确复杂度/精度降级 | [TKDE DOI](https://doi.org/10.1109/TKDE.2020.3036524) |
| RRCF | `rrcf.py:1-10,58-60,157-164` 均匀选维并以 inverse depth 打分；论文按维度 range 比例抽样并用 displacement/codisp。属于 paper-name proxy | [ICML 2016 paper](https://proceedings.mlr.press/v48/guha16.html) |
| SOD | `sod.py:98-107` 方差分母固定使用配置 `self.ref_set`；`:122-142` 小样本时实际 reference set 会缩小但分母不变，得到错误小样本分数 | [SOD paper](https://imada.sdu.dk/u/zimek/publications/PAKDD2009/pakdd09-SOD.pdf) |
| DBSCAN anomaly | `dbscan.py:48-130` 把聚类算法改成 distance-to-core novelty score；实现可以有用，但不是 DBSCAN 原论文的 anomaly score，应标 adaptation | [KDD 1996 paper](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) |
| SSIM template/map/struct | `ssim.py`、`ssim_map.py`、`ssim_struct.py` 把图像质量指标改成 `1-SSIM` 模板异常分数/图；方向合理，但原论文不是 anomaly detector，fidelity 为空 | [SSIM paper](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) |

ABOD、COF、COPOD、ECOD、IForest、LDOF、LOF、MCD、Sampling 的核心公式/方向在本次静态与固定输入检查中未发现 material divergence；OCSVM 与 PyOD 的 rank 差来自本地预标准化、`nu=contamination`、`gamma='scale'` 等默认参数，不是方向错误。KNN 默认 `largest` 对应 Ramaswamy kth-distance，但 registry 只写 `SIGMOD 2000`，引用不够明确。

### M-17 — 深度模型 checkpoint / map 契约缺口

| 模块 | 结论与本地证据 | 论文算法判断 |
|---|---|---|
| EfficientAD | `efficientad.py:248-275,645-683` 始终返回网络输入大小 256×256 的 map，丢弃原图尺寸；官方补充材料要求再 bilinear resize 回 original size | PDN/AE/三损失、70k schedule、quantile/fusion 对齐；保留 `paper-adaptation`，披露 fixed-size map |
| PatchCore | `patchcore.py:245-312` 保存 `_n_neighbors_fit`，load 却不恢复实际 scoring 使用的 `self.n_neighbors`；以 n=2 保存、n=1 constructor 加载会改变 Eq.(7) reweighting 分支 | 算法路径对齐；save/load capability 是 partial，而不是 paper-fidelity 缺陷 |
| PaDiM | `padim.py:159-237` 未保存/验证 backbone、image/resize size、covariance 与 preprocessing contract；payload 虽保存 `patch_shape`、并以 feature-index 数量间接校验 `d_reduced`，却不验证保存 moments/grid 与当前 extractor/input contract；可把 32px 工件加载进 64px 实例，之后推理失败 | 显式 `pretrained=True` 时核心随机通道、位置 Gaussian、Mahalanobis、sigma4 对齐；checkpoint contract 不完整 |
| SPADE | `spade.py:425-438,496-503` 把保存机器的 `device` 当模型语义恢复；CUDA 工件在 CPU-only 机器可能无法加载 | 论文平方距离路径对齐；device 应是 load-time runtime 参数 |

EfficientAD 主源：[WACV 2024 supplemental](https://openaccess.thecvf.com/content/WACV2024/supplemental/Batzner_EfficientAD_Accurate_Visual_WACV_2024_supplemental.zip)。PatchCore 工件对照：[official save/load](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/patchcore.py#L234-L273)。

### M-18 — ALAD 默认训练缺少论文的 validation early-stop / best-checkpoint protocol

`pyimgano/models/alad.py:231-275` 固定最多 100 epochs，却不接收论文协议中的 validation set；基类默认 `early_stopping_patience=None`，即使手动启用也监控训练总 loss 且不恢复 validation feature score 最佳权重（`pyimgano/models/base_deep.py:527-532,594-611`）。论文 Table IX 给出 patience=10；作者发布路径要求 early stop，并在验证 feature score 改善时保存 checkpoint。

主源：[ALAD paper](https://arxiv.org/pdf/1812.02288)，[fixed author validation/checkpoint path](https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection/blob/1f1c3109c957bdfab23d684638124282beee7894/alad/run.py#L467-L495)。本地 GAN objective、EMA 与 L1 feature score 在已核范围基本一致；这是训练/权重选择协议偏差，维持 `paper-adaptation`，但 implementation status 应披露 `no-paper-validation-early-stop`。

### M-19 — AST 默认冻结随机 EfficientNet，而不是论文的 ImageNet-pretrained backbone

`pyimgano/models/ast.py:453-471` 默认 `pretrained_backbone=False`，`:68-82` 随后冻结全部 backbone 参数。论文 §4.2.1 要求 ImageNet-pretrained EfficientNet-B5；作者代码也直接 `EfficientNet.from_pretrained`。因此默认 AST 的特征空间不是 paper profile，只有显式开启预训练才进入论文配置边界。

主源：[AST paper](https://openaccess.thecvf.com/content/WACV2023/papers/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.pdf)，[fixed author model](https://github.com/marco-rudolph/AST/blob/8c243ad9adac68e874f87edc6618aa5ea2827228/model.py#L41-L56)。现有测试甚至断言默认 `False` 并使用 tiny fake extractor，只固化了 offline default，没有验证 paper semantics。

### M-20 — DeepSVDD 图像入口是 objective proxy，默认缺少论文实验的 CNN/AE 初始化

`pyimgano/models/deep_svdd.py:49-78,142-188,237-254` 默认使用通用 bias-free MLP、StandardScaler、`use_autoencoder=False`；vision 路径在 `:406-416` 只是把 32×32 原始像素展开。论文 Eq.(3)–(5) objective、每 5 epochs 更新 radius 的正文路径在本地基本正确，但论文图像实验使用 LeNet-style CNN、global contrast/min-max preprocessing 和 DCAE encoder 初始化；作者发布入口也默认先做 AE pretraining。

主源：[Deep SVDD paper](https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf)，[fixed author training entry](https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/1901612d595e23675fb75c4ebb563dd0ffebc21e/src/main.py#L25-L51)，[fixed CIFAR network](https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/1901612d595e23675fb75c4ebb563dd0ffebc21e/src/networks/cifar10_LeNet.py#L8-L89)。因此 `core_deep_svdd` 是 objective adaptation，`vision_deep_svdd` 应明确标 paper proxy，不能把 smoke test 理解为原图像实验复现。

### M-21 — FCDD 默认不是论文的 pretrained/frozen MVTec backbone

`pyimgano/models/fcdd.py:115-143,170-172` 默认 `pretrained=False`、`freeze_features=False`。论文 MVTec profile 使用 ImageNet-pretrained VGG11，并冻结前部 feature layers；作者的默认 FCDD CNN224 VGG 路径也加载 ImageNet 权重并冻结 features。local loss、network head、confetti synthesis 与 Gaussian receptive-field map 在已核范围未见同级偏差，但默认 backbone 足以阻断论文结果复现。

主源：[FCDD paper](https://openreview.net/pdf?id=A5VV3UyIQz)，[fixed author VGG implementation](https://github.com/liznerski/fcdd/blob/4fa850215792b5f4a3405151e30127a5b67dc3b6/python/fcdd/models/fcdd_cnn_224.py#L61-L127)。现有 alignment test 使用 `pretrained=False, freeze_features=True`，实际冻结的是随机权重，并没有检查 constructor 默认或论文 backbone。

### M-22 — MemAE 对 float `[0,1]` 图像执行了错误的二次缩放

`pyimgano/models/memae.py:263-276,339-356` 在训练和推理都无条件执行 `float()/127.5-1`。`uint8 [0,255]` 会正确映射到 `[-1,1]`，但合法的 float `[0,1]` 会被压到约 `[-1,-0.992]`；两种等价输入因此产生不同网络 batch。作者数据路径是先 `ToTensor()` 再 `Normalize(0.5,0.5)`。

主源：[MemAE paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)，[fixed author preprocessing](https://github.com/donggong1/memae-anomaly-detection/blob/ceece7714fb241e82ef3f3785d3d1ed86c28113e/script_training.py#L64-L74)。本地论文 2D topology、memory addressing 与 objective 在已核范围基本一致；修复前应注明 `uint8-only preprocessing semantics`。论文还把 test reconstruction error 缩放到 `[0,1]`，本地 raw MSE rank 相同但绝对 threshold 不可直接对照。

## 完整模块结论矩阵

<!-- AUDIT_MATRIX_START -->

下面按实现模块合并共享 constructor 的 aliases；“对齐”仍只表示已核对范围内未发现 material divergence，不表示跑出了论文指标。`H/M/C` 指向上面的详细证据。

### A. Classical / feature algorithms

| 模块（主要 registry aliases） | 审计状态 | 结论边界 |
|---|---|---|
| `abod`（`core_abod`, `vision_abod`） | 对齐 | 核心公式、方向和固定数据检查未发现偏差 |
| `cblof`（`core_cblof`, `vision_cblof`） | 缺陷 | 非法簇分割被静默兜底，venue metadata 错；M-16 |
| `cof`（`core_cof`, `vision_cof`） | 对齐 | 核心链距/方向未发现偏差 |
| `copod`（`core_copod`, `vision_copod`） | 对齐 | 核心经验 copula 路径未发现偏差 |
| `dbscan`（`core_dbscan`, `vision_dbscan`, `dbscan_anomaly`） | 明确适配、metadata 待补 | distance-to-core novelty extension，不是原论文 anomaly score；M-16 |
| `ecod`（`core_ecod`, `vision_ecod`） | 对齐 | 核心 ECDF/tail aggregation 未发现偏差 |
| `feature_bagging`（core/vision 及 `_spec` aliases） | 声明过强 | 子空间上界和 score combination 偏离论文；M-16 |
| `hbos`（`core_hbos`, `vision_hbos`） | 缺陷 | 训练范围外查询被 clip 到边缘高密度 bin；M-16 |
| `hst`（`core_hst`, `vision_hst`） | 声明过强 | inverse-leaf-mass proxy，缺论文的工作空间/双窗口质量；M-16 |
| `iforest`（`core_iforest`, `vision_iforest`） | 对齐 | sklearn IsolationForest backend 与方向翻转正确 |
| `imdd`, `lmdd`（core/vision aliases） | 缺陷 | 不使用训练分布、查询 batch/调用次数依赖；H-03 |
| `inne`（`core_inne`, `vision_inne`） | 条件对齐、有边界缺陷 | 默认数据路径未见偏差；`max_samples=1` 全 NaN；M-16 |
| `knn`（`core_knn`, `vision_knn`） | 对齐、引用待明确 | 默认 kth-distance 对应 Ramaswamy；当前只写 `SIGMOD 2000` |
| `ldof`（`core_ldof`, `vision_ldof`） | 对齐 | 核心 LDOF 公式/方向未发现偏差 |
| `lid`（`core_lid`, `vision_lid`） | 缺陷 | MLE 缺倒数且排序口径反转；H-02 |
| `loci`（`core_loci`, `vision_loci`） | API 缺陷 | dataset-level 方法被暴露成不稳定 inductive scorer；H-03 |
| `loda`（`core_loda`, `vision_loda`） | 缺陷 | score 多除一次投影数；M-09 |
| `lof_core`, `lof_native` / `lof`（core/vision aliases） | 对齐 | sklearn novelty 与 native wrapper 的核心方向未发现偏差 |
| `loop`（`core_loop`, `vision_loop`） | 缺陷 | nPLOF 用 centered std 而非 zero-mean RMS；M-01 |
| `lscp`（core/vision 及 `_spec` aliases） | 明确适配、metadata 待补 | local region 未按作者实现先标准化；M-16 |
| `mcd`（`core_mcd`, `vision_mcd`） | 对齐 | sklearn FastMCD/Mahalanobis 路径未发现偏差 |
| `ocsvm`（`core_ocsvm`, `vision_ocsvm`） | 条件对齐 | 方向正确；预标准化、`nu=contamination`、`gamma='scale'` 是本地默认适配 |
| `odin`（`core_odin`, `vision_odin`） | 声明过强 | OOS 用邻居平均 indegree，不是原 dataset-level 定义；M-16 |
| `padim_lite`（core/vision aliases） | 明确适配 | metadata 已如实标 `image-level-gaussian-proxy` |
| `patchcore_lite`, `patchcore_online`（core/vision aliases） | 明确适配 | metadata 已如实标 image-level proxy / online variant |
| `pca`（`core_pca`, `vision_pca`） | 缺陷、错误归属 | 默认全成分重构退化；实现不是所引 ICDM 2003 classifier；H-05 |
| `qmcd`（`core_qmcd`, `vision_qmcd`） | 缺陷、错误归属 | 极远 OOS 分数方向错误；均匀设计论文不是该 detector 的直接来源；H-06 |
| `rgraph`（`core_rgraph`, `vision_rgraph`） | 声明过强 | kNN/RBF proxy，论文参数大量未使用；M-16 |
| `rod`（`core_rod`, `vision_rod`） | 明确降级、metadata 待补 | 高维时只抽 256 个三元子空间；M-16 |
| `rrcf`（`core_rrcf`, `vision_rrcf`） | 声明过强 | 随机树与 inverse-depth proxy，不是论文 range/codisp；M-16 |
| `sampling`（`core_sampling`, `vision_sampling`） | 对齐 | 核心采样分数与方向未发现偏差 |
| `sod`（`core_sod`, `vision_sod`） | 边界缺陷 | 小样本缩小 reference set 后分母不随之更新；M-16 |
| `sos`（`core_sos`, `vision_sos`） | API 缺陷 | 查询 batch 内重建 SOS，单样本恒 0；H-03 |
| `suod`（core/vision 及 `_spec` aliases） | 声明过强 | 只是 score ensemble，缺 SUOD acceleration system；M-06 |
| `ssim`, `ssim_map`, `ssim_struct`（4 个 template/map/struct aliases） | 明确适配、metadata 待补 | 把图像质量指标改成模板 anomaly score/map；M-16 |
| `extra_trees_density`（core/vision aliases） | 缺陷、非论文 baseline | 分母错误使用查询 batch 大小；H-04 |
| 其余无直接论文 claim 的 classical modules | 不适用 paper fidelity | `cook_distance`, `cosine_mahalanobis`, `dcorr`, `dtc`, `elliptic_envelope`, `gmm`, `kde`, `kde_ratio`, `kmeans`, `knn_cosine`, `knn_cosine_calibrated`, `knn_degree`, `kpca`, `mad`, `mahalanobis`, `mahalanobis_shrinkage`, `mst_outlier`, `neighborhood_entropy`, `pca_md`, `random_projection_knn`, `rzscore`, `score_ensemble`, `score_standardizer`, `studentized_residual`；已反查是否冒充论文复现，未作 paper-fidelity 背书 |

### B. Native deep implementations

| 模块（主要 registry aliases） | 审计状态 | 结论边界 |
|---|---|---|
| `alad`（`vision_alad`） | 明确适配、训练协议偏差 | 核心 score 路径基本一致；默认缺论文 validation early-stop/best-checkpoint selection；M-18 |
| `ast`（`vision_ast`） | 默认配置偏差 | RGB network/loss/schedule 基本一致；默认冻结随机 EfficientNet，须显式启用论文预训练；M-19 |
| `cflow`（`vision_cflow`） | API 缺陷、默认配置偏离论文 | per-call 全局归一化导致 score batch-dependent；默认关闭论文 encoder 预训练；H-10/M-07 |
| `cutpaste`（`cutpaste`, `vision_cutpaste`） | 对齐（论文边界） | ResNet18、3-way objective、schedule 和 Gaussian score 未见偏差；无作者官方代码，未验证论文指标 |
| `deep_svdd`（core/vision aliases） | objective adaptation / 图像 proxy | Eq.(3)–(5) 基本一致；默认 MLP/32px 展平、无 AE pretraining，不是论文 CNN 实验；M-20 |
| `devnet`（`devnet`, `vision_devnet`） | 检测路径对齐、明确缺 localization | signed top-K/deviation loss 基本一致；未实现论文 Eq.(9) 像素图，metadata 已披露 |
| `dfm`（`vision_dfm`） | 明确适配 | Gaussian branch broadly 对齐；class-conditional OOD 改为 one-class industrial |
| `differnet`（`differnet`, `vision_differnet`） | 条件对齐 | detection path 与固定作者源码对照未见偏差；无 gradient localization 已披露 |
| `draem`（`vision_draem`） | 明确适配、声明需收窄 | 网络/损失/schedule 接近论文，synthetic anomaly 分布明显简化；M-02 |
| `efficientad`（2 aliases） | 明确适配、有 map 契约缺口 | 核心网络/损失/score 未见同级偏差；输出不恢复原图尺寸；M-17 |
| `fastflow`（`vision_fastflow`） | 明确适配 | ResNet stages/flow objective 对齐已核范围；概率归一化等是本地 API 选择 |
| `fcdd`（`vision_fcdd`） | 默认配置偏差 | loss/head/map 基本一致；默认不使用/冻结论文 pretrained VGG；M-21 |
| `memae`（`vision_memae`） | 预处理缺陷、明确适配 | 2D architecture/objective 基本一致；float `[0,1]` 被二次按 255 缩放；M-22 |
| `padim`（`padim`, `vision_padim`） | 条件对齐、artifact 不完整 | `pretrained=True` 时核心随机通道/位置 Gaussian/Mahalanobis 未见偏差；默认关闭论文预训练，checkpoint config 不封闭；M-07/M-17 |
| `panda`（`vision_panda`） | 论文路径对齐、API 缺陷 | PANDA-Early 核心路径未见偏差；`predict()` 返回连续 score；M-15 |
| `patchcore`（`vision_patchcore`） | 条件对齐、artifact 不完整 | `pretrained=True` 时 backbone/coreset/Eq.(7) 未见偏差；默认关闭论文预训练，checkpoint 未恢复实际 `n_neighbors`；M-07/M-17 |
| `patchcore_lite_map`（`vision_patchcore_lite_map`） | 明确适配 | metadata 已标 lite patch-memory proxy，不冒充完整 PatchCore |
| `reverse_distillation`（2 aliases） | 对齐 | teacher/OCBE/decoder/cosine loss-map/defaults 与固定作者源码未见 material divergence |
| `riad`（`riad`, `vision_riad`） | 明确适配 | mask/network/loss/ensemble broadly 对齐；无作者代码，精确实现与指标未证实 |
| `simplenet`（`vision_simplenet`） | 条件对齐、source-divergent | `pretrained=True` 时符合论文列出的主配置；默认关闭预训练，作者发布代码又用另一组配置；C-02 |
| `softpatch`（`vision_softpatch`） | 条件对齐 | `pretrained=True` 时 LOF hard removal、weighted memory、score/map 与固定作者源码未见 material divergence；默认关闭论文预训练；M-07 |
| `spade`（`spade`, `vision_spade`） | 条件对齐、artifact 有缺口 | `pretrained=True` 时 squared L2/pyramid 路径与论文 source 一致；默认关闭论文预训练，公开二次实现另有距离分歧，checkpoint 错误持久化 device；C-03/M-07/M-17 |
| `stfpm`（`vision_stfpm`） | 条件对齐、source-divergent | `pretrained_teacher=True` 时 map/objective 更接近论文；默认关闭预训练，作者发布代码的 map/checkpoint selection 又与论文不同；C-01 |
| `visionad`（`vision_visionad`） | 方法路径对齐、source-divergent | Eq.(1)–(6)、memory/cosine/view/top-1% 路径基本一致，且拒绝随机 backbone；论文默认与作者发布 layer/profile 有分歧，无真实权重 golden |

### C. VLM and external adapters

| 模块（主要 registry aliases） | 审计状态 | 结论边界 |
|---|---|---|
| `aaclip`（`vision_aaclip`） | 缺陷、条件适配 | image score 遗漏官方 pixel/image fusion；另受 OpenCLIP layout 条件影响；H-01/H-07 |
| `adaclip`（`vision_adaclip`） | 条件对齐 | native method path 未见同级偏差；真实正确性取决于 OpenCLIP 版本/layout；H-01 |
| `anomalydino`（`vision_anomalydino`） | 方法适配、artifact 不完整 | 448/DINOv2-S14、mask/cosine/top-k 默认路径对齐；保存配置不自包含；M-11 |
| `bayesianpf`（`vision_bayesianpf`） | 外部 facade、声明需收窄 | 只转调用户 backend，不能独立加载官方 checkpoint；M-08 |
| `filopp`（`vision_filopp`） | 外部代理、artifact 校验缺陷 | legacy registry key 的兼容边界已披露；`strict=False` 不检查 key coverage；M-04 |
| `glad`（`vision_glad`） | 方法适配、数值未证实 | preset/ADS/SAFF 可映射；diffusers 重构路径无真实 checkpoint E2E；M-14 |
| `inctrl`（`vision_inctrl`） | 条件对齐 | 方法结构未见同级偏差；真实 OpenCLIP layout/version 仍是 H-01 前置条件 |
| `logsad`（`vision_logsad`） | 外部代理 | 固定官方 source 的推理 delegate；本仓库只验证 adapter，不背书 upstream 工件/指标 |
| `oneformore`（`vision_oneformore`） | 外部代理、有 score 缺陷 | map/reconstruction adapter 边界基本诚实；image score 漏 AP smoothing；H-08 |
| `patchcore_inspection_backend`（1 alias） | 外部代理、artifact 契约缺口 | 未采用/验证官方保存的 `input_shape`，pickle 仅可信来源；M-05 |
| `promptad`（`vision_promptad`） | 条件对齐 | prompt method path 未见同级偏差；真实 OpenCLIP tokenizer/layout 受 H-01/M-10 影响 |
| `realnet`（`vision_realnet`） | 明确适配 | AFS/reconstruction/RRS 可映射；训练使用 static preblended pairs 而非在线 SDAS；M-03 |
| `regad`（`vision_regad`） | 方法适配、协议未复现 | STN/SimSiam/Gaussian map 对齐已核范围；未复现多轮 benchmark/oracle selection；M-14 |
| `univad`（`vision_univad`） | 外部代理 | 固定官方 source 的推理 delegate；本仓库不证明 upstream checkpoint 数值 |
| `winclip`（2 aliases） | 独立复原、条件未证实 | 无作者官方代码/checkpoint可核；temperature 未由论文给定，且受 H-01 影响；M-13 |
| `anomalib_backend`（22 aliases） | 外部代理、身份未验证 | `vision_anomalib_checkpoint` 及 CFA/CFlow/CSFlow/DFKDE/DFM/Dinomaly/DRAEM/DSR/EfficientAD/FastFlow/FRE/Ganomaly/PaDiM/PatchCore/Reverse-Distillation/RKDE/STFPM/SuperSimpleNet/UFlow/VLM-AD/WinCLIP aliases 都进入同一 `TorchInferencer`；alias 不验证 checkpoint 实际模型；M-12 |

### D. Deep modules without paper claims

| 模块 | 审计状态 | 结论边界 |
|---|---|---|
| `openclip_backend`, `openclip_patch_map` | 非论文 baseline，但实现有缺陷 | 三个 OpenCLIP prompt/patch/KNN entries 的 `not-applicable` 标注诚实；仍受 layout、tokenizer 与 ignored `knn_index` 问题影响；H-01/M-10 |
| `ae`, `ae1svm`, `dst`, `favae`, `gcad`, `industrial_wrappers`, `memseg`, `ref_patch_distance`, `superad`, `torch_autoencoder`, `vae`, `vqvae` | 不适用 paper fidelity | Registry 已标 generic baseline/pipeline；本轮反查没有把它们误列成论文复现，也不对其性能作论文背书 |

<!-- AUDIT_MATRIX_END -->

## 测试、工具与可重复证据

### 已通过

- `python tools/audit_registry.py`：通过 registry introspection。
- `python tools/audit_pixel_map_models.py`：通过 pixel-map tag/method 一致性检查。
- registry 中 36 个唯一 `paper_url` 自动 GET：36/36 返回 HTTP 200。
- 本报告 66 个唯一外部 citation URL 再检查：无 404 或 transport hard failure；少数 DOI/publisher endpoint 对自动客户端返回 202/403，未把它们误记为正文可抓取证据。
- `tools/audit_third_party_notices.py`、`tools/audit_no_reference_clones_tracked.py`、`tools/audit_repo_links.py` 均通过；但它们的检测边界见 H-09。

这些结果证明内部 contract 基本稳定，不证明论文 fidelity。H-01 至 H-10 都能在上述相关测试通过的同时存在。

### 固定数值回归

运行：

```bash
python tools/repro_paper_audit.py
```

脚本固定所有数组、seed、Torch eval mode 和 dropout，且不联网、不加载预训练权重；覆盖 H-01/H-02/H-03/H-04/H-05/H-06/H-08 及 HBOS 边界示例。审计完成后它已改为断言修复后的不变量；后文 finding 中的旧缺陷数值来自审计基线。当前关键结果为 OpenCLIP layout 差异 `0.0`、LID `1.4426950408889634`、LOCI/SOS/IMDD singleton-context 差异 `0.0`、IMDD repeat 差异 `0.0`、ExtraTrees 重复查询差异 `0.0`，且 QMCD 两侧远点均高于中心。当前环境为 Python 3.10.14、NumPy 1.26.4、SciPy 1.14.1、scikit-learn 1.7.2、Torch 2.13.0+cu130；跨版本应优先比较脚本表达的不变量，而不是要求所有浮点末位 bitwise 相同。

### 当前合同门禁

- `python -m pyimgano.pyim_cli --audit-metadata --json`：exit 0；279 entries 中 required、recommended、invalid-field issues 均为 0。
- `python tools/audit_score_direction.py`：exit 0；64 个非深度 `core_*` 模型全部满足固定合成 outlier 的平均分高于 normal，warning 0。
- `python tools/repro_paper_audit.py`：exit 0；固定公式、layout、query-context、重复调用与边界不变量全部通过。

### 可选依赖与官方权重端到端

本次最终环境实际安装并执行了 `onnx==1.22.0`、`onnxruntime==1.23.2`、`onnxscript==0.7.1`、`diffusers==0.40.0`、`open_clip_torch==3.3.0`、`anomalib==2.6.0`、`peft==0.20.0`；与 `torch==2.13.0+cu130`、`torchvision==0.28.0+cu130`、`transformers==5.16.1`、`huggingface-hub==1.29.0` 共同通过全套测试。这里记录的是已验证矩阵，不宣称所有上下游版本的任意组合都兼容。

- 原先因缺少 ONNXScript、diffusers、OpenCLIP、anomalib 而跳过的 6 个测试已全部真实执行；聚焦集为 `9 passed`。
- OpenCLIP 使用官方 `ViT-B-32 / laion2b_s34b_b79k` 权重，在 RTX 3070 Ti 上完成图像 patch、模型专用 tokenizer、文本 prompt、image score 与 `2×224×224` anomaly map 全链路；所有输出 finite。固定合成输入中 normal/defect score 为 `0.0160531 / 0.0167220`。
- PatchCore 使用 torchvision 官方 ImageNet `wide_resnet50_2` 权重，在同一 GPU 上完成 fit、coreset、kNN score 与 `224×224` anomaly map；memory bank 为 `31×1024`，固定合成输入中 normal/defect score 为 `6.12838 / 26.10154`，map finite。
- 上述合成缺陷结果只证明真实权重、CUDA、预处理、特征、评分与像素图接口连通，不替代 MVTec AD/VisA 上的 AUROC/AUPRO 论文数值复现。

### Semgrep

使用 `p/python` 与 `p/security-audit` 的 200 条规则，以 `--no-git-ignore` 扫描 1083 个目标文件：exit 0，finding 0，parse coverage 约 100%。基线发现的 pickle trusted-boundary、动态 URL/SSRF 与相关 helper 问题均已核销；Semgrep 是模式扫描门禁，不等于跨函数安全形式化证明。

### 依赖与静态安全

- `python -m pip check`：依赖闭包完整，无 broken requirements。
- `pip-audit`：完整可选 profile 仅报告 Lightning 2.6.5 的 `PYSEC-2026-3624` / `CVE-2026-58659`；截至本报告日期，上游 2.6.5 仍无已发布修复版，跟踪见 `https://github.com/Lightning-AI/pytorch-lightning/issues/21913`。CI 只 allowlist 这一个 ID，新增 finding 会失败。
- Bandit `-ll`：medium/high finding 均为 0；Semgrep 结果见上节。
- 所有内建 checkpoint 恢复默认 fail-closed；非执行 JSON/NumPy 安全格式和安全 Torch state-dict 可直接恢复，旧 pickle/joblib 需要显式 trust 决策。

### 全套测试

<!-- FULL_TEST_RESULT_START -->
在安全刷新后的完整可选依赖环境中从零启动运行 `pytest -q`：`2887 passed in 1055.02s (0:17:35)`，exit 0；收集总数为 2887，0 skipped、0 warnings、0 failed。PyTorch 2.13 的 ONNX 导出路径已迁移到 opset 18 与 `dynamic_shapes`，旧 opset 请求走显式兼容路径，不再产生迁移 warning。

相比审计基线，本轮新增了手算公式、query-batch invariance、双 OpenCLIP layout、官方 image-score reducer、checkpoint schema/config、原图 map 尺寸、metadata profile、示例下载授权透传和跨环境 optional-dependency 模拟等回归门禁。当前 OpenCLIP 3.3.0 与官方大权重已做单版本端到端；旧版 OpenCLIP 和完整 benchmark/version matrix 仍属于外部验证边界。
<!-- FULL_TEST_RESULT_END -->

## 基线修复优先级记录（均已核销）

以下列表保留审计时的修复顺序，实际核销结果见报告开头的当前工作树状态与 finding 清单。

1. 先修 H-01：pin/限定 OpenCLIP 版本，并基于明确 capability/version 选择 layout；优先使用官方 `forward_intermediates` / `output_tokens`，增加真实旧版与新版集成测试。
2. 修复 H-02/H-04/H-05/H-06 与 LoOP 公式，并新增给定手算输入的 golden tests；测试必须断言数值，不只断言 finite/shape。
3. 对 LOCI/SOS/IMDD/LMDD 明确选择：实现有训练状态的 novelty extension，或把 API 标为 transductive 并拒绝任意 OOS singleton scoring。
4. 为 AA-CLIP/One-for-More 的官方 image-score 路径增加 regression fixture；对 STFPM 则明确选“论文正文”还是“作者发布代码”为规范来源，并把 source divergence 写进 metadata。
5. 为 DRAEM/RealNet/ALAD/AST/DeepSVDD/FCDD 建立显式 paper reproducibility profile，分离 offline-safe constructor；修 MemAE float 输入归一化，不再用一个宽泛 `paper-adaptation` 掩盖关键训练/预处理差异。
6. 对外部 checkpoint 记录 upstream commit、预处理、schema hash、模型 key coverage，并拒绝不匹配工件。
7. 补 PyOD BSD-2-Clause notice/归属，逐个审查 ported files；把 third-party audit 从 marker-presence 改成来源清单/哈希/历史可追踪检查。
8. 补齐 canonical paper title、DOI/URL、official repo、fidelity、known deviations；CI 至少要求直接论文 claim 有这些字段。

## 限制

- 没有在 MVTec AD/VisA 等完整 benchmark 上重训全部深度模型；因此不声称论文指标可复现或不可复现。
- 需要作者 checkpoint、外部仓库或可选依赖的模型，仅对本地 adapter、工件契约和可核实路径下结论。
- “对齐”只表示本次审计未发现 material divergence；不等于形式化证明，也不等于第三方认证。
- 涉及确定源码差异的关键证据尽量使用固定 commit 和具体行；少数只列 repository root 的链接仅用于 provenance 或说明尚未完成固定工件 E2E，不能据此理解为已冻结全部上游依赖。
