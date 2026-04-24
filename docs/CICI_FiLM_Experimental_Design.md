# CICI-FiLM: Clinical-Imaging Conditioned Inpainting with Feature-wise Linear Modulation

## 实验设计与开发说明书 (Experimental Design & Development Document)

> **命名释义**：**CICI-FiLM** 表示一种**临床-影像联合条件驱动的修补生成框架**（*Clinical-Imaging Conditioned Inpainting with Feature-wise Linear Modulation*），其通过输出层 FiLM 调制将患者级条件向量注入预训练 inpainting 网络，在尽量保持既有 backbone 与 pipeline 不变的前提下，实现 COPD 病灶纹理的个体化生成。

| 项目 | 信息 |
|------|------|
| **项目** | COPD Digital Twin Lung — 个性化建模升级 |
| **方案代号** | CICI-FiLM |
| **核心定位** | 将无条件纹理填充升级为临床-影像条件驱动的个性化病灶合成 |
| **设计原则** | 最高性价比 · 最低代码侵入 · 最快跑通实验 |

## 实验背景（Experimental Background）

本实验属于 **COPD Digital Twin Lung** 项目的个性化病灶建模补充研究，
其定位是在既有 texture synthesis / inpainting 主线已完成的基础上，针对患者异质性建模不足所开展的增量式实验，
而非脱离现有系统重新构建的一条独立技术路线。

现有流程已能够基于健康 Atlas、患者级病灶 Mask 与预训练生成器合成形态上合理的 COPD 病灶纹理，
并在 PatchGAN 基线上取得稳定结果；然而，该流程本质上仍以通用生成机制为主，
对患者个体差异的利用主要停留在空间掩膜约束层面，尚未显式引入能够表征病灶严重度与纹理统计特征的患者级条件信息。
因此，当前系统虽然具备病灶生成能力，但在病理统计量层面的个体化控制上仍存在不足。

基于这一背景，本次补充实验的目标是在**尽量不破坏现有预训练 backbone 与既有 pipeline** 的前提下，
以 PatchGAN 作为唯一实验基线，通过两阶段增量式改造——Phase ① 自适应 HU 校准与 Phase ② 冻结 backbone 的 Output-level FiLM 微调——
将患者级临床-影像联合信息编码为 5 维条件向量（c₁~c₅），从而实现对 COPD 病灶严重度及其 HU 统计特征的个体化调制。

本实验拟验证的核心命题是：**显式条件驱动是否能够在保持原有生成质量不退步的同时，提高合成 COPD CT 与真实患者 COPD CT 在 EI、病灶 HU 分布及相关统计量上的一致性**。
换言之，本研究关注的并非另起炉灶地追求新的生成主干，而是在现有系统的可比框架内，检验患者级条件建模是否能够为个体化病灶合成提供可量化、可归因的增益。

因此，本节的实验定位为后续方法学设计、消融链构建及个体化建模叙事提供统一的研究动机：
即在保持基线系统工程稳定性的前提下，将 COPD Digital Twin Lung 从"通用病灶生成"推进到"显式条件驱动的个体化病灶生成"。

---

## 目录

1. [方案总论与学术定位](#1-方案总论与学术定位)
2. [特征工程设计](#2-特征工程设计)
3. [网络结构改造剖析](#3-网络结构改造剖析)
4. [训练策略与预期效果](#4-训练策略与预期效果)
5. [系统级验证与评估体系](#5-系统级验证与评估体系)
6. [实施路线与代码改动清单](#6-实施路线与代码改动清单)

---

## 1. 方案总论与学术定位

### 1.1 问题陈述

当前 COPD Digital Twin 系统采用 **"共享健康 Atlas + 患者级病灶 Mask + 无条件 Inpainting"** 的生成范式。
该范式的核心瓶颈在于推理阶段 `inference_fuse.py` 第 461–464 行：

```python
target_stats = {
    'mean': -965.0,   # 固定先验
    'std': 45.0       # 固定先验
}
```

这意味着 **所有 29 位 COPD 患者**（不论 GOLD Ⅰ 级轻度还是 GOLD Ⅳ 级极重度）的生成病灶，
最终都被强制校准到同一个 HU 分布。这在物理上等价于：

> 让一位 FEV1/FVC=0.68 的轻度患者和一位 FEV1/FVC=0.28 的极重度患者，
> 拥有完全相同的肺气肿纹理密度和对比度。

该固定先验直接导致了当前 PatchGAN 在验证集上 ΔEI = 3.49% 的误差下限无法突破。

### 1.2 解决方案概述

**CICI-FiLM** 通过两个层级的改造解决上述问题：

| 层级 | 改造内容 | 训练成本 |
|------|---------|---------|
| **阶段 ①** 推理期后处理 | 将 hardcoded `target_stats` 替换为每位患者的真实病灶 HU 统计 | **零** |
| **阶段 ②** 条件化微调 | 在冻结的预训练 PatchGAN backbone 输出端添加 FiLM 调制层，注入 **5 维**条件向量 | **~50 epochs** |

> **基线模型选择说明**：两个阶段均以 **PatchGAN** 为唯一实验基线（验证集 ΔEI=3.49%，五模型中最优综合性能）。
> 采用"单最优基线聚焦"策略，将全部实验资源集中用于消融链设计，而非横向重复多模型。
> 详见 §4.4 节的学术合理性论证。

### 1.3 学术创新点

1. **条件化范式升级**：将 COPD 患者 GOLD 临床分级与影像学定量特征
   （EI、病灶 HU 统计）融合为统一的 **5 维**条件向量，注入 3D Inpainting 生成网络，
   首次实现从"无条件通用生成"到"临床-影像双重条件化个性化生成"的范式跃升。
2. **极致轻量化**：采用 Output-level FiLM + Frozen Backbone 策略，
   仅训练 **2,434** 个参数（0.010% / 22,587,123 总参数）即实现严重度可控生成。
3. **增量式消融链**：Exp-0 → Exp-1 → Exp-2 构成严格递进的消融实验，
   每一步的独立增益可量化，且每步的 baseline 均为已发表结果（无重训风险）。
4. **特征精简哲学**：通过严格的信息冗余审计，从 22 维候选特征精简至 5 维，
   以 Output-level FiLM 的物理自由度（γ, β 两个标量）为约束上界，
   避免了小样本下条件空间稀疏性问题（23^(1/5) ≈ 1.9 有效采样点/维）。

---

## 2. 特征工程设计

### 2.1 条件向量设计原则

CICI-FiLM 的条件化目标是控制生成病灶的密度与纹理——物理上等价于调节 Output-level FiLM 的
两个自由度：**β（整体偏移，对应 HU 均值）** 和 **γ（缩放系数，对应 HU 对比度）**。

特征选择遵循三条原则：

1. **物理映射直接性**：特征须能直接约束 γ 或 β 的目标值，
   而非通过多跳间接推断（排除 FEV1/FVC、mMRC、CAT 等功能学或症状学指标）。
2. **信息正交性**：各特征在 Output-level FiLM 的低维作用空间内不冗余
   （排除与 c₁+c₃ 高度共线的 FEV1/FVC，r ≈ 0.85–0.92）。
3. **工程零侵入性**：所有特征均为标量，无需改变网络输入通道数（`in_channels=1` 不变），
   从而保证全部预训练权重 100% 兼容
   （排除需要 3D map 输入的空间距离图、叶标签等）。

在 23 例训练集的约束下，5 维是信息完备性与条件空间采样密度之间的帕累托最优点：
每维约有 23^(1/5) ≈ 1.89 个有效采样点，增加维度会使条件空间迅速稀疏化。

### 2.2 五维特征的临床-物理语义映射

**特征 c₁: 全肺肺气肿指数 (Global EI)**

- **物理含义**：EI 是肺气肿最权威的定量影像学指标，定义为肺实质中 HU < −950
  体素占总肺实质体素的比例。EI 越高，意味着更大比例的肺组织已经发生气肿性破坏。
- **临床关联**：EI 与 GOLD 分级高度相关（Pearson r ≈ 0.7–0.85，文献 Madani et al., Radiology 2006），
  是 COPD 严重度的影像学金标准。
- **对生成的控制作用**：直接决定网络需要生成多大比例的"极低密度"体素。
  EI = 0.05 的轻度患者只需少量 HU < −950 区域，
  EI = 0.40 的重度患者需要大面积极低密度填充。
- **计算公式**：
  ```
  EI = Σ 𝟙[CT(v) < -950 ∧ LungMask(v) > 0] / Σ 𝟙[LungMask(v) > 0]
  ```
- **数据来源**：`copd_XXX_clean.nii.gz` + `copd_XXX_mask.nii.gz`



**特征 c₂: 病灶体积占比 (Lesion Volume Ratio)**

- **物理含义**：肺气肿病灶（emphysema mask 标记区域）占全肺体积的百分比。
  与 EI 的区别在于——EI 基于 HU 阈值（<−950）实时计算，
  而 lesion_vol_ratio 基于已分割的 emphysema mask，两者高度正相关但不完全等价
  （mask 可能包含 −950 < HU < −910 的轻度低衰减区域）。
- **临床关联**：反映病灶的"空间占据程度"，是放射科报告中常用的定量描述。
- **对生成的控制作用**：与 EI 形成互补约束——EI 控制密度，vol_ratio 控制范围。
- **计算公式**：
  ```
  lesion_vol_ratio = Σ 𝟙[EmphMask(v) > 0] / Σ 𝟙[LungMask(v) > 0]
  ```
- **数据来源**：`copd_XXX_emphysema.nii.gz` + `copd_XXX_mask.nii.gz`

**特征 c₃: 病灶区 HU 均值 (Lesion HU Mean)**

- **物理含义**：所有被标记为肺气肿的体素在原始 CT 中的平均 HU 值。
  典型取值范围为 −980 ~ −940 HU。
  值越低（如 −978）表示气肿破坏越严重（更接近空气 −1000 HU）；
  值越高（如 −942）表示气肿尚处于早期。
- **临床关联**：肺气肿亚型不同，HU 分布不同——
  小叶中心型（centrilobular）通常均值略高于全小叶型（panlobular），
  因为前者混合了正常实质。
- **对生成的控制作用**：**直接替代当前 hardcoded 的 `target_stats['mean'] = -965.0`**。
  这是阶段 ① 自适应校准的核心改造目标。
- **计算公式**：
  ```
  lesion_HU_mean = mean(CT[EmphMask > 0])
  ```
- **数据来源**：`copd_XXX_clean.nii.gz` + `copd_XXX_emphysema.nii.gz`

**特征 c₄: 病灶区 HU 标准差 (Lesion HU Std)**

- **物理含义**：病灶区域 HU 分布的离散程度。
  标准差大（如 55 HU）意味着病灶内部密度变化显著（混合型/不均匀气肿）；
  标准差小（如 25 HU）意味着病灶密度高度均匀（弥漫性均匀气肿）。
- **对生成的控制作用**：**直接替代当前 hardcoded 的 `target_stats['std'] = 45.0`**。
  控制生成纹理的"对比度"——高 std 意味着纹理需要更多内部变异。
- **计算公式**：
  ```
  lesion_HU_std = std(CT[EmphMask > 0])
  ```
- **数据来源**：同 c₃

**特征 c₅: GOLD 分级 (GOLD Stage)**

- **临床含义**：GOLD（Global Initiative for Chronic Obstructive Lung Disease）分级
  是 COPD 严重度的国际标准分类体系，基于支气管扩张剂后 FEV1 占预计值的百分比：
  - GOLD Ⅰ（轻度）：FEV1 ≥ 80% pred
  - GOLD Ⅱ（中度）：50% ≤ FEV1 < 80% pred
  - GOLD Ⅲ（重度）：30% ≤ FEV1 < 50% pred
  - GOLD Ⅳ（极重度）：FEV1 < 30% pred
- **编码方式**：序数编码 `GOLD / 4`，映射为 {0.25, 0.5, 0.75, 1.0}。
  选择序数编码而非 one-hot 的原因是：GOLD 分级具有**内在有序性**
  （Ⅳ 严格重于 Ⅲ），序数编码保留了这一先验知识，
  且仅占 1 维而非 4 维，与极简设计原则一致。
- **对生成的控制作用**：作为"疾病严重度的语义锚点"，在功能层面为模型提供
  临床分期先验；在实验层面支撑 GOLD 分组实验（Exp-R2）。
- **与 c₁~c₄ 的关系**：c₅ 与 c₁(EI) 存在中度相关（r ≈ 0.7-0.85），
  但两者信息不等价——EI 是连续影像学量化值，GOLD 是离散临床判断标准。
  保留 c₅ 的理由同时兼顾两点：其一，4 个离散分级构成的序数锚点能为 CondEncoder
  提供额外的分组约束，辅助小样本下的跨患者泛化；其二，GOLD 是论文叙事中
  "临床条件驱动"的最具公认度的指标，支撑 Exp-R2 GOLD 分组实验的可视化展示。

> **已移除的临床特征**：FEV1/FVC、mMRC、CAT 经信息冗余分析后不纳入——
> 三者与 c₁~c₄ 高度共线或与像素级纹理生成任务正交，
> 无法为 Output-level FiLM 的 γ/β 提供有效增量信息。
> 用户仅需提供 `data/clinical_data.csv`（`patient_id` + `GOLD` 两列）。

### 2.3 条件向量定义与归一化

```python
# 条件向量 c ∈ R^5
c = [
    c1,  # global_EI         ∈ [0, 1]                  影像学 — FiLM β 全局上下文
    c2,  # lesion_vol_ratio  ∈ [0, 1]                  影像学 — FiLM γ 范围上下文
    c3,  # lesion_HU_mean    → z-score → sigmoid        影像学 — β 的直接目标值
    c4,  # lesion_HU_std     → z-score → sigmoid        影像学 — γ 的直接目标值
    c5,  # GOLD_ordinal      ∈ {0.25, 0.5, 0.75, 1.0}  临床   — 分期锚点
]
```

### 2.4 归一化策略

| 特征 | 原始范围 | 归一化方法 | 归一化后范围 | 实现细节 |
|------|---------|-----------|------------|---------|
| c₁ global_EI | [0, ~0.5] | 直接使用 | [0, 1] | 比率值，天然归一化 |
| c₂ lesion_vol_ratio | [0, ~0.5] | 直接使用 | [0, 1] | 同上 |
| c₃ lesion_HU_mean | [−990, −930] HU | z-score → sigmoid | [0, 1] | μ, σ 从训练集 23 例计算 |
| c₄ lesion_HU_std | [20, 70] HU | z-score → sigmoid | [0, 1] | 同上 |
| c₅ GOLD | {1, 2, 3, 4} | x / 4 | {0.25, 0.5, 0.75, 1.0} | 保留有序性 |

c₃ 和 c₄ 选择 z-score + sigmoid 而非 min-max：sigmoid 对异常值具有天然鲁棒性，
可将极端值压缩至 (0,1) 边界附近而不溢出，且不依赖预设的值域上下界。
归一化参数（μ, σ）在 23 例训练集上计算并固定，推理时复用。

> **数据获取**：c₁~c₄ 由 `scripts/extract_patient_features.py` 自动从 CT 和
> emphysema mask 中批量计算；c₅ 由用户提供 `data/clinical_data.csv`
>（仅需 `patient_id` 和 `GOLD` 两列）。

### 2.5 与原有空间特征（Lesion Mask）的协作关系

5 维标量条件向量是"密度/纹理约束"，原有的患者级病灶 Mask 是"空间位置约束"，
两者正交互补，共同构成完整的个性化生成条件。

当前系统的空间个性化能力来自 `data/03_mapped/copd_XXX/copd_XXX_warped_lesion.nii.gz`，
在推理时用于挖空（`inference_fuse.py` L301-303）和 mask-only 回写（L400）。
**这两步逻辑完全不改动**，5 维条件向量仅在此基础上叠加"生成什么样的纹理"的约束。

| Lesion Mask 提供 | 5 维条件向量提供 |
|:---|:---|
| 病灶的**空间位置**（哪些体素需要填充） | 病灶的**密度目标**（β 控制 HU 均值偏移） |
| 病灶的**空间范围**（mask 的 3D 形态） | 病灶的**对比度目标**（γ 控制 HU 方差） |
| 病灶的**解剖上下文**（周围正常组织） | 病灶的**临床分期语义**（GOLD 锚点） |

> **Lesion Mask = "在哪里画"，Condition Vector = "画成什么样"**

---

## 3. 网络结构改造剖析

### 3.0 核心架构概念定义

本节对 CICI-FiLM 方法所依赖的三个核心术语给出精确的学术定义，
说明它们如何共同构成"零侵入式个性化改造"的完整机制。

#### 3.0.1 Wrapper（包装器模式）

**定义**：Wrapper（包装器）是一种软件设计模式，指将一个已有对象（被包装对象）完整嵌入新对象内部，
在不修改被包装对象任何内部代码的前提下，通过对外暴露新接口来扩展其功能。

在本项目中，`ConditionedGenerator` 将预训练的 `InpaintingUNet`（backbone）作为 `self.backbone` 子模块持有，
backbone 的所有参数名、参数形状、计算逻辑**一字不改**：

```python
class ConditionedGenerator(nn.Module):
    def __init__(self, backbone, cond_dim=5):
        super().__init__()
        self.backbone = backbone   # 持有原始模型，不修改任何内部结构
        self.cond_encoder = ...    # 新增：条件编码器
        self.film = ...            # 新增：FiLM 调制层
```

**为什么 Wrapper 能保证预训练权重 100% 兼容？**

PyTorch 的 `load_state_dict()` 通过**参数名（key）精确匹配**来恢复权重。
预训练检查点中的 136 个 key（如 `encoder.0.conv.conv.0.weight`）
在 `ConditionedGenerator` 中变为 `backbone.encoder.0.conv.conv.0.weight`——
仅添加了 `backbone.` 前缀，而 backbone 本身依然可以用原始 key 独立加载：

```python
# 第一步：backbone 单独加载，key 完全匹配
backbone = InpaintingUNet()
backbone.load_state_dict(ckpt['generator_state_dict'])  # 136个key，零警告

# 第二步：包裹进 Wrapper，新增的 FiLM 参数随机初始化
model = ConditionedGenerator(backbone, cond_dim=5)
# model.backbone.* → 来自预训练（冻结）
# model.cond_encoder.* / model.film.* → 新增（可训练）
```

这意味着无论现有有多少实验结果，只要加载检查点的方式不变，所有历史结果均可 100% 复现。

#### 3.0.2 FiLM（Feature-wise Linear Modulation，特征线性调制）

**定义**：FiLM 是 Perez et al.（AAAI 2018）提出的一种条件化机制，
通过从条件信息中预测一组**仿射变换参数**（缩放系数 γ 和偏移系数 β），
对神经网络某一层的特征图做逐通道线性调制：

$$\text{FiLM}(h \mid \mathbf{c}) = \gamma(\mathbf{c}) \odot h + \beta(\mathbf{c})$$

其中 $h$ 是被调制的特征张量，$\gamma(\mathbf{c})$ 和 $\beta(\mathbf{c})$ 由条件向量 $\mathbf{c}$ 经小型网络（CondEncoder）投影得到。

**物理含义（以本项目为例）**：

- **β（偏移）** → 控制生成纹理的整体 HU 基准线。β < 0 使所有体素 HU 值整体下移（更暗，更接近空气），
  对应重度气肿患者需要更低 HU 的物理现实。
- **γ（缩放）** → 控制生成纹理的"内部对比度"。γ > 1 放大各体素间的 HU 差异（更高方差），
  对应非均匀气肿（如小叶中心型）的纹理多样性。

**与传统条件化（直接 Concat/Add）的区别**：FiLM 的调制参数由 CondEncoder 动态预测，
同一个 γ/β 对整张特征图生效，因此**对特征图的全局统计量有精确控制**，
这与我们的目标（控制 HU 均值和方差）完全对齐。

**零初始化策略（identity start）**：

γ_proj 和 β_proj 的权重/偏置全部初始化为零：

```python
nn.init.zeros_(self.gamma_proj.weight); nn.init.zeros_(self.gamma_proj.bias)
nn.init.zeros_(self.beta_proj.weight);  nn.init.zeros_(self.beta_proj.bias)
```

初始状态下 γ = 0（实际缩放因子 = γ + 1 = 1），β = 0，FiLM 输出 = 原始输入。
这保证了**训练起点与 Exp-0 baseline 完全等价**，ΔEI 不会因引入 FiLM 而初始变差。

#### 3.0.3 Output-level（输出层级调制）

**定义**：Output-level FiLM 指将 FiLM 调制施加在 backbone **最终输出**（而非中间解码器特征层）的策略。
与之对应的是 Multi-level FiLM，后者在多个中间特征层上注入调制。

**与 Multi-level FiLM 的对比**：

| 维度 | Multi-level FiLM | Output-level FiLM（本方案） |
|:-----|:-----------------|:--------------------------|
| 注入位置 | 多个中间特征层（例如解码器 4 层） | 最终输出层（通道数 = 1） |
| 实现前提 | 在当前代码库中需暴露中间特征，通常意味着修改 `forward()` 或引入 Hook | 仅依赖最终输出，保持纯 Wrapper 实现 |
| 新增 FiLM 投影参数 | 若按 4 层 `256/128/64/32`、`cond_emb_dim=64` 计，约 **62,400** 个投影参数 | **130** 个投影参数 |
| 与 backbone 耦合 | 高：依赖中间层通道数与层级结构 | 低：仅依赖输出通道数 |
| 调制能力 | 更适合层级特征重分布 | 更适合最终输出全局统计量控制 |

**本项目选择 Output-level 的物理合理性**：

对于输出通道数 = 1 的 3D Inpainting 网络，Output-level FiLM 在体素级别等价于：

```text
y'(v) = (γ + 1) · y(v) + β
```

其中 y(v) 是 backbone 在体素 v 处的输出值（归一化后的 HU）。这在物理上恰好等价于
**条件驱动的全局亮度/对比度调制**——与 c₃（直接决定 β 目标值）和 c₄（直接决定 γ 目标值）
的物理语义完全对应。换言之，Output-level 并非表达力不足的妥协，
而是与当前任务目标（控制全局 HU 统计量）在物理上的**精确匹配**。

对于更复杂的空间差异控制（如更强地干预中间层特征分布或局部纹理形成），
Multi-level FiLM 理论上具有更强的层级调制能力；但这会显著增加结构耦合、参数量与变量控制难度，
超出了本项目以"最小侵入方式控制最终病灶统计量"为核心的设计目标。

#### 3.0.4 三者结合：零侵入式个性化改造的完整机制

三个概念的组合形成了一条清晰的工程设计链：

```
① Wrapper 模式
   保证预训练权重 100% 兼容，历史实验结果零影响
        ↓
② Output-level 定位
   无需访问内部结构，FiLM 只操作最终输出，维持 Wrapper 纯洁性
        ↓
③ FiLM 调制
   从 5 维条件向量动态预测 (γ, β)，对生成结果做全局亮度/对比度调整
        ↓
结果：用 2,434 个新增参数（0.010% 参数量），在不改动任何已有代码的前提下，
      实现从"一刀切"无条件生成到"临床-影像双重条件驱动"的个性化范式升级
```

**对审稿人的一句话概括**：

> "We freeze the pretrained backbone and attach a lightweight FiLM branch at the output level
> via a wrapper pattern, requiring no modification to any existing module while enabling
> patient-specific conditioning through only 2,434 trainable parameters."

---

### 3.1 现有模型适用性评估

项目中 `network.py` 定义了 6 种 Generator 架构。
基于 `checkpoints/*/best.pth` 实际检查点验证的参数量如下：

| 模型 | Generator 参数量 | Discriminator | Best Epoch | Best Val Loss |
|------|:----------------:|:---:|:---:|:---:|
| InpaintingUNet | 22,587,123 | — | 117 | 0.0256 |
| PartialConvUNet | 22,587,609 | — | 142 | 0.0306 |
| PatchGAN (UNet+D) | 22,587,123 | 11,051,460 | 148 | 0.0453 |
| AttGAN (AttUNet+D) | 23,005,156 | 11,051,460 | 127 | 0.0294 |
| MAE-PatchGAN | 22,587,123 | 11,051,460 | 53 | 0.0642 |
| DDPM | — | — | — | 灾难过拟合 |

所有 Generator 的 `forward()` 签名均为 `forward(self, x)`，
只接受一个形状为 `(B, 1, D, H, W)` 的张量输入。

**适用性结论**：

| 评估维度 | 结论 |
|---------|------|
| **架构兼容性** | ✅ 所有 Generator 均可作为 CICI-FiLM 的 backbone，因为 FiLM 在输出端调制，不侵入内部结构 |
| **权重兼容性** | ✅ Wrapper 模式下 backbone 的 `state_dict` key 完全不变，可直接 `load_state_dict()` |
| **推荐优先级** | PatchGAN > AttGAN > UNet（按现有实验 ΔEI 表现排序） |
| **Discriminator** | 阶段 ② 微调时不使用 Discriminator（冻结 backbone，只训练 FiLM 的 ~2,500 参数，不需要对抗训练） |

### 3.2 Wrapper + Output-level FiLM 的确定

#### 3.2.1 Wrapper 模式的工程意义

CICI-FiLM 采用 `ConditionedGenerator(backbone)` 的包装式实现：预训练生成器作为
`self.backbone` 被完整保留，新增模块仅包括 `cond_encoder` 与 `film` 两个条件分支。
该设计的核心价值在于**零侵入性**：backbone 的参数名、参数形状、前向计算图和推理接口均保持不变，
因此原始检查点可先按既有流程加载到 backbone，再由 wrapper 在外部叠加条件化能力。
在工程上，这一策略同时满足三个目标：

1. **检查点兼容**：`generator_state_dict` 可直接恢复到原始 backbone，历史结果可复现；
2. **代码稳定**：`network.py` 与现有推理路径无需改写，新增逻辑局限于外层模块；
3. **回退安全**：当 `condition=None` 时，wrapper 退化为原始生成器，条件化升级不会破坏既有系统行为。

因此，Wrapper 在本项目中并非单纯的软件封装技巧，而是保证"最小代码改动 + 最大实验可比性"的核心机制。

#### 3.2.2 Output-level FiLM 的任务匹配性

条件调制施加在 backbone 最终输出上，形成如下映射：

```text
y'(v) = (γ(\mathbf{c}) + 1) \cdot y(v) + β(\mathbf{c})
```

其中 `y(v)` 为 backbone 输出，`γ(\mathbf{c})` 与 `β(\mathbf{c})` 由 5 维条件向量经 CondEncoder 预测。
该形式在物理上等价于对生成病灶做**全局 HU 偏移与对比度缩放**：β 控制密度中心，γ 控制纹理方差。
这与本项目的条件目标完全一致——c₃（lesion_HU_mean）直接约束 β，c₄（lesion_HU_std）直接约束 γ，
c₁/c₂/c₅ 提供严重度、范围与临床分期上下文。

Output-level FiLM 的意义不在于追求更复杂的特征干预，而在于以**最小参数量、最小结构耦合**
实现对病灶统计量的可解释控制。对于以 EI、病灶 HU 均值和标准差为核心评价目标的 COPD 个体化生成任务，
这种单次全局 affine 调制与任务物理目标是严格一致的。

### 3.3 ConditionedGenerator 详细设计

#### 3.3.1 计算图

```
输入:  x ∈ R^{B×1×D×H×W}        条件: c ∈ R^{B×5}
         │                            │
         │                            │
         ▼                            ▼
  ┌──────────────┐             ┌────────────┐
  │   Backbone   │             │ CondEncoder│
  │ (PatchGAN    │             │  5 → 32    │
  │  Generator,  │             │  32 → 64   │
  │  完全冻结)    │             │  (ReLU)    │
  └──────┬───────┘             └─────┬──────┘
         │                           │
    y ∈ R^{B×1×D×H×W}          e ∈ R^{B×64}
         │                      ┌────┴────┐
         │                      │         │
         │                ┌─────┴──┐  ┌───┴────┐
         │                │ γ_proj │  │ β_proj │
         │                │ 64→1   │  │ 64→1   │
         │                └───┬────┘  └───┬────┘
         │                    │           │
         │               γ ∈ R^{B×1}  β ∈ R^{B×1}
         │                    │           │
         │                    └─────┬─────┘
         │                          │
         ▼                          ▼
  ┌─────────────────────────────────────────┐
  │  FiLM: y' = (γ + 1) ⊙ y + β            │
  │  (γ+1 保证初始状态 ≈ identity)           │
  └──────────────────┬──────────────────────┘
                     │
                y' ∈ R^{B×1×D×H×W}
                     │
                     ▼
                  输出
```

注意 `γ + 1` 的设计：FiLM 的 γ_proj 权重和 bias 初始化为零，
因此初始 γ = 0，γ + 1 = 1，β = 0，
此时 y' = 1 · y + 0 = y，
即 **wrapper 在训练初期的行为等价于无条件 backbone**。
这保证了从预训练权重 warm-start 时不会出现性能骤降。

#### 3.3.2 参数量分析

| 组件 | 参数计算 | 参数量 | 可训练 |
|------|---------|:------:|:-----:|
| Backbone (PatchGAN Generator) | 已有，冻结 | 22,587,123 | ❌ 冻结 |
| CondEncoder Linear(**5**→32) | **5**×32 + 32 | **192** | ✅ |
| CondEncoder Linear(32→64) | 32×64 + 64 | 2,112 | ✅ |
| γ_proj Linear(64→1) | 64×1 + 1 | 65 | ✅ |
| β_proj Linear(64→1) | 64×1 + 1 | 65 | ✅ |
| **总计** | | **22,589,557** | **2,434 (0.010%)** |

> **说明**：相比原 8 维方案（2,530 参数），5 维方案减少 96 个参数（−3.8%）。
> 减少量本身微乎其微，但**减少 3 维输入**降低了条件空间稀疏性，
> 有效采样密度从 23^(1/8)≈1.47 提升至 23^(1/5)≈1.89 点/维（提升 29%）。

#### 3.3.3 完整实现代码

新建文件 `src/04_texture_synthesis/conditioned_model.py`：

```python
"""
CICI-FiLM: 条件化生成模型

Output-level FiLM Wrapper — 在已有 backbone 输出端添加条件驱动的
Feature-wise Linear Modulation，实现病灶严重度可控生成。

架构特点:
  - backbone (InpaintingUNet / AttentionUNet) 完全不改、权重完全复用
  - 仅在最终输出层做一次 channel-wise affine 调制
  - 训练时冻结 backbone，只训练 FiLM 参数 (2,434 / 22.5M = 0.010%)
  - condition=None 时自动退化为无条件模式（与原模型行为一致）

使用方法:
    backbone = InpaintingUNet()
    backbone.load_state_dict(pretrained_weights)
    model = ConditionedGenerator(backbone, cond_dim=5)
    model.freeze_backbone()
    output = model(x, condition)  # condition: (B, 5) or None
"""
import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation for single-channel output"""

    def __init__(self, cond_emb_dim, num_channels=1):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_emb_dim, num_channels)
        self.beta_proj  = nn.Linear(cond_emb_dim, num_channels)
        # 初始化: γ≈0 (加上 residual +1 后 → 1), β≈0
        # → 初始 FiLM 行为 = identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h, cond_emb):
        gamma = self.gamma_proj(cond_emb)  # (B, C)
        beta  = self.beta_proj(cond_emb)   # (B, C)
        # reshape for broadcasting: (B,C) → (B,C,1,1,1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1.0
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta


class ConditionedGenerator(nn.Module):
    """
    条件化生成器 Wrapper

    包裹任意已有 3D Inpainting backbone，
    在其输出端添加 FiLM 调制。
    """

    def __init__(self, backbone, cond_dim=5, cond_emb_dim=64):
        super().__init__()
        self.backbone = backbone
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, cond_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.film = FiLMBlock(cond_emb_dim, num_channels=1)

    def freeze_backbone(self):
        """冻结主干网络，只训练条件分支"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[CICI-FiLM] Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.3f}%)")

    def forward(self, x, condition=None):
        y = self.backbone(x)
        if condition is not None:
            cond_emb = self.cond_encoder(condition)
            y = self.film(y, cond_emb)
        return y
```

### 3.4 代码改动对现有模块的影响

| 现有模块 | 是否修改 | 说明 |
|---------|:-------:|------|
| `network.py` (InpaintingUNet) | ❌ | 不动一行 |
| `network.py` (AttentionUNet) | ❌ | 不动一行 |
| `network.py` (PatchDiscriminator) | ❌ | 阶段②不使用 |
| `network.py` (create_model) | ❌ | backbone 仍通过原工厂函数创建 |
| `losses.py` (InpaintingLoss) | ❌ | 损失函数不变 |
| `dataset.py` (LungPatchDataset) | ⚠️ 微调 | `__init__` 加 `conditions` 参数；`__getitem__` 返回多一个字段 |
| `train.py` (Trainer) | ⚠️ 微调 | `train_epoch`/`validate` 中传递 condition |
| `inference_fuse.py` (fuse_lesion) | ⚠️ 微调 | 函数签名加 `patient_condition`；target_stats 自适应 |
| `run_phase3_pipeline.py` | ⚠️ 微调 | 推理循环加载 JSON 并传参 |
| **新增** `conditioned_model.py` | 🆕 | ConditionedGenerator + FiLMBlock |
| **新增** `extract_patient_features.py` | 🆕 | 批量计算影像特征 |

---

## 4. 训练策略与预期效果

### 4.1 两阶段实施方案

#### 阶段 ① — 零训练自适应后处理

**改动本质**：将 `inference_fuse.py` 中 hardcoded 的 `target_stats = {'mean': -965.0, 'std': 45.0}`
替换为从 `data/patient_features.json` 读取每位患者的真实 `lesion_HU_mean` 和 `lesion_HU_std`。

**精确改动位置**：

| 文件 | 行号 | 改动 |
|------|------|------|
| `inference_fuse.py` | L246 | 函数签名新增 `patient_condition=None` |
| `inference_fuse.py` | L459-464 | `if patient_condition: target_stats = {...}` |
| `run_phase3_pipeline.py` | L1035 前 | 加载 patient_features.json |
| `run_phase3_pipeline.py` | L1048-1056 | fuse_lesion 调用新增 `patient_condition=...` |

**物理机制**：

当前 hardcoded 方案对所有患者执行相同的 z-score 匹配：

```
y_calibrated = (y_AI - μ_AI) × (σ_target / σ_AI) + μ_target
其中 μ_target = -965, σ_target = 45 (固定)
```

改造后变为：

```
μ_target = patient_features[pid]['lesion_HU_mean']  # 如 copd_001: -972
σ_target = patient_features[pid]['lesion_HU_std']   # 如 copd_001: 38
```

这样 GOLD Ⅰ 的轻度患者（mean ≈ -945, std ≈ 30）和 GOLD Ⅳ 的重度患者
（mean ≈ -978, std ≈ 55）会被校准到各自对应的 HU 分布，而非统一的 -965/45。

#### 阶段 ② — FiLM 条件化微调（冻结主干）

**训练流程**：

```python
# 1. 创建 backbone 并加载预训练权重（以 PatchGAN 为例）
from src.04_texture_synthesis.network import create_model
backbone, _ = create_model('patchgan')  # 返回 (generator, discriminator)

import torch
pretrained = torch.load('checkpoints/patchgan/best.pth', map_location='cpu')
backbone.load_state_dict(pretrained['generator_state_dict'])
# → 22,587,123 参数全部加载，136 个 key 完美匹配

# 2. 包裹为条件化模型
from src.04_texture_synthesis.conditioned_model import ConditionedGenerator
model = ConditionedGenerator(backbone, cond_dim=5)
model.freeze_backbone()
# → 打印: [CICI-FiLM] Trainable: 2,434 / 22,589,557 (0.010%)

# 3. 只训练 FiLM 参数，50 epochs
trainer = Trainer(model, discriminator=None, config=config)
trainer.train(train_loader, val_loader, epochs=50,
              checkpoint_dir='checkpoints/patchgan_cici')
```

**关键细节**：

- `discriminator=None`：阶段②不使用对抗训练。
  原因：只有 2,434 个可训练参数，对抗训练会导致 mode collapse 风险。
  仅用 L1 + Perceptual + HU Constraint 损失足够。
- `epochs=50`：因为只训练 2,434 个参数，收敛极快。
  23 例 × 50 patches/例 = 1,150 patches × 50 epochs = 57,500 次梯度更新。
  对于 2,434 个参数，这已经是充分训练。

### 4.2 基线模型选择原则

后续个性化实验仅在 PatchGAN 上开展。该决策基于三点：

1. **最优基线原则**：PatchGAN 在前期五模型比较中取得最佳验证集 ΔEI（3.49%），
   因此最适合作为条件化升级的承载 backbone；
2. **单一变量控制**：本工作的核心贡献是三步消融链
   （Exp-0 → Exp-1 → Exp-2）中阶段①与阶段②的独立增益，
   因而需要所有实验共享同一 backbone 以保证内部可比性；
3. **方法通用而实验聚焦**：CICI-FiLM 的 wrapper 实现与具体 backbone 解耦，
   但本文的实验目标是验证条件化机制，而非穷举其在多架构上的迁移性。

因此，单模型聚焦并非缩减实验范围，而是为保证消融链完整性与因果归因清晰性所做的实验设计选择。

### 4.3 冻结主干微调 FiLM 的科学性

Phase ② 采用冻结 PatchGAN backbone、仅训练 FiLM 条件分支的策略。其合理性来自
**统计安全性、变量控制能力与归因完备性**三方面的一致支持。

#### 4.3.1 统计安全性

| 方案 | 可训练参数 | 有效训练样本 | 参数/样本比 |
|:-----|:----------:|:----------:|:----------:|
| **冻结主干 + FiLM 微调** | **2,434** | 23 例 × ~50 patch = 1,150 | **2.1** |
| 从头重训（含主干） | 22,589,557 | 1,150 | **~19,643** |

在 23 例训练集下，从头重训的参数/样本比高出三个数量级，
其最优解更可能是对训练样本及其条件组合的记忆，而非学习稳定的条件响应映射。
相反，FiLM 微调只在既有纹理先验之上学习低维条件到 `(γ, β)` 的映射，
统计风险处于可控范围内。

#### 4.3.2 变量控制能力

冻结 backbone 后，Phase ② 的唯一新增学习对象是 5 维条件分支，
训练轮次、初始化状态、主干收敛程度与损失权衡均继承自既有 PatchGAN 检查点。
这将实验自由度压缩到单一变量——**条件信息是否被注入，以及其如何通过 FiLM 起作用**。
相对地，从头重训会同时改变初始化、收敛路径、超参数敏感性与训练时长，
从而破坏 Exp-0、Exp-1 与 Exp-2 之间的严格可比性。

#### 4.3.3 提升归因的闭环逻辑

本工作的核心不是证明"重训后性能可以更好"，而是证明
**性能提升来自临床-影像条件化本身**。为此，三步消融链必须满足单一变量控制原则：

- **Exp-0 → Exp-1**：唯一变化是 `target_stats` 从固定值替换为患者真实 c₃/c₄，
  因而性能差异可归因于阶段①自适应 HU 校准；
- **Exp-1 → Exp-2**：backbone 保持冻结，唯一新增可训练部分为 2,434 参数的 FiLM 条件分支，
  因而性能差异可归因于 5 维条件向量及其调制机制。

在这一设计下，阶段②的任何增益都不再受额外训练轮次、随机初始化或主干重新收敛的干扰。
这使得 CICI-FiLM 的核心命题——"显式条件驱动带来个体化增益"——能够形成闭环验证。
从头重训则无法完成这一因果隔离，因此不适合作为本研究的主实验策略。

#### 4.3.4 方法学结论

据此，冻结主干微调 FiLM 不是工程上的折中方案，而是当前样本规模、任务目标与实验逻辑下
**唯一同时满足可训练性、可比较性与可归因性**的方法。该策略保证了：

1. backbone 保留既有 COPD 纹理生成先验；
2. 条件分支仅学习低维个体化调制；
3. Exp-2 相对 Exp-1 的性能变化可以被严格解释为条件化机制的贡献。

### 4.4 预期指标变化（更新版）

| 指标 | Exp-0 (Baseline, PatchGAN) | Exp-1 (+ 自适应校准) | Exp-2 (+ FiLM 微调, 5d) |
|------|:-:|:-:|:-:|
| **ΔEI (%) ↓** | 3.49 | 1.5 ~ 2.5 | 1.0 ~ 1.5 |
| **PSNR (dB)** | baseline | ≈ 不变 | ≈ 不变 (backbone 冻结) |
| **SSIM** | baseline | ≈ 不变 | ≈ 不变 (backbone 冻结) |
| **HU KL ↓** | baseline | 显著降低 | 进一步降低 |
| **条件响应 r** | — | — | > 0.85 (EI Sweep 实验) |
| **GOLD 组间差异** | n.s. | 弱显著 | 显著 (p < 0.05) |
| **可训练参数** | 22,587,123 | 0（无训练） | **2,434 (5d CondEncoder)** |
| **训练时间** | 已完成 | **0** | **~50 min** |

**估计依据**：

- ΔEI 降低的主要贡献来自阶段①（hardcoded target_stats 是误差的主要来源）；
- 阶段②在阶段①基础上进一步改善，但上限受 output-level FiLM（全局 affine，2 自由度）约束；
- PSNR/SSIM 不退步：backbone 冻结保证初始 = baseline，FiLM 从 identity 出发。

---

## 5. 系统级验证与评估体系

### 5.1 实验设计总览

```
消融实验链（主干，每步仅改变一个变量）:
  Exp-0 (Baseline, 无条件，PatchGAN epoch 148)
    └─ Exp-1 (+ 自适应 HU 校准, 零训练)
        └─ Exp-2 (+ FiLM 条件化微调, 5 维条件向量, 50 epochs)

条件响应验证（证明条件向量确实有效）:
  Exp-R1: EI Sweep 实验 (连续严重度控制验证)
  Exp-R2: GOLD 分级分组实验 (离散临床分期验证)

特征消融（5 维版本，验证各维度独立贡献）:
  Exp-A0: 无条件基线（同 Exp-0）
  Exp-A1: 仅影像学特征 (c₁~c₄, 4 维)
  Exp-A2: 仅临床特征 (c₅ GOLD, 1 维)
  Exp-A3: 全部 5 维特征（同 Exp-2）
```

**评估参考集**：全部实验均在验证集 **copd_024~029（6 例）** 上运行，
"ref CT" 统一指代对应患者的**真实 COPD CT**（`data/01_cleaned/copd_clean/copd_XXX_clean.nii.gz`）。
详见 §5.2 的参考集定义说明。

### 5.2 指标体系

#### 5.2.0 参考集定义与评估范式说明

> **本节是理解全部指标的前提，请务必在论文 Evaluation 节中明确陈述。**

**"ref CT" 的精确定义**：

本文档中所有指标公式里出现的 `ref`、`ref CT`、`EI_ref` 均统一指代：

> **同一患者对应的真实 COPD CT**（`data/01_cleaned/copd_clean/copd_XXX_clean.nii.gz`，
> 验证集 copd_024~029，共 6 例）

**注意**：`ref CT` 是真实 COPD 病理 CT，而非健康 Atlas（`atlas_XXX.nii.gz`）。
这一区分对于正确理解 ΔEI 等指标的意义至关重要。

---

**核心评估范式：统计-生理等价性（Statistical-Physiological Equivalence）**

生成模型（Inpainting）的评估与重建模型（Reconstruction）有本质区别：

| 评估范式 | 重建模型（如超分辨率、去噪） | 本项目生成模型 |
|:--------|:------------------------|:-------------|
| **目标** | 像素级还原（pixel-wise fidelity） | 分布级等价（distribution-level equivalence） |
| **核心指标** | PSNR / SSIM（越高越好） | ΔEI / HU KL（分布匹配度） |
| **ref 的角色** | Ground truth，生成结果应尽量等于 ref | 参考分布，生成结果的统计量应与 ref 的统计量对齐 |
| **PSNR 的意义** | 主要指标 | **次要约束指标**（只需不退步，因为不同患者间的像素差异是预期的） |

因此，在本项目的评估框架中：

- **ΔEI、HU KL、HU Mean/Std Error** 是**主要指标**——衡量生成 CT 的气肿严重度统计量
  是否与同一患者真实 COPD CT 的统计量对齐（即"生成出来的气肿程度对不对"）
- **PSNR / SSIM** 是**基线保护指标**——确保 CICI-FiLM 的改造没有损害整体图像质量，
  不要求超过 Atlas-only baseline（像素级差异在患者间是合理的）

> **写给论文 Evaluation 节的英文表述**：
>
> "We adopt a **statistical-physiological equivalence** framework rather than
> pixel-wise reconstruction metrics. The key evaluation criterion is whether
> the fused CT's emphysema index (EI) and lesion HU distribution match those
> of the same patient's real COPD CT (copd_XXX_clean.nii.gz), not whether
> the fused image is pixel-identically close to it. PSNR and SSIM serve as
> non-regression constraints to ensure overall image quality is preserved."

#### 5.2.1 影像学质量约束指标（非退步保证）

| 指标 | 计算方法 | 参考（ref） | 要求 |
|------|---------|:----------:|------|
| **PSNR** | `10 × log10(MAX² / MSE)` | 同患者真实 COPD CT | ≥ Exp-0 baseline（不退步） |
| **SSIM** | 结构相似性（全卷） | 同患者真实 COPD CT | ≥ Exp-0 baseline（不退步） |
| **MAE**（病灶区） | `mean(abs(fused[mask] - ref[mask]))` | 同患者真实 COPD CT | ≤ Exp-0 baseline |
| **Non-lesion L1** | `max(abs(fused[mask=0] - atlas[mask=0]))` | **健康 Atlas**（非病灶区不变） | = 0（mask-only 覆盖，不改非病灶区） |

> **注意**：Non-lesion L1 的参考是 Atlas（确保非病灶区像素完全不变）；
> 其余指标的参考均为**真实患者 COPD CT**。两类参考必须严格区分。

#### 5.2.2 个性化精度指标（核心创新验证——主要指标）

以下指标衡量生成 CT 在**统计-生理等价性**维度的个性化精度，
即生成 CT 的病灶区统计量与同一患者真实 COPD CT 的匹配程度：

| 指标 | 公式 | 参考（ref） | 物理含义 |
|------|------|:----------:|---------|
| **ΔEI** | `abs(EI_fused - EI_ref) / (EI_ref + 1e-10) × 100%` | 真实患者 COPD CT | 气肿严重度的相对匹配误差 |
| **HU KL 散度** | `KL(P_fused ‖ P_ref)`（mask 区域 HU 直方图） | 真实患者 COPD CT | 病灶区整体 HU 分布的匹配质量 |
| **HU 均值误差** | `abs(mean(fused[mask]) - mean(ref[mask]))` | 真实患者 COPD CT | 密度中心匹配精度（β 调制效果） |
| **HU 标准差误差** | `abs(std(fused[mask]) - std(ref[mask]))` | 真实患者 COPD CT | 纹理对比度匹配精度（γ 调制效果） |

> **指标层级说明**：HU 均值误差和 HU 标准差误差分别直接评估 FiLM 的 β 和 γ
> 两个自由度的调制效果，与 §2.2 中 c₃/c₄ 的物理含义形成完整的闭环验证。

#### 5.2.3 条件响应性指标（证明"条件有效"）

| 指标 | 计算方法 | 预期值 |
|------|---------|--------|
| **Condition-EI Correlation** | `Pearson(c1_input, EI_measured)` | r > 0.85 |
| **Cross-Patient Variance Ratio** | `var(EI_conditioned) / var(EI_unconditioned)` | > 1.0 |
| **GOLD-EI Gradient** | ANOVA / Kruskal-Wallis: EI ~ GOLD group | p < 0.05 |

### 5.3 实验协议详解

#### 5.3.1 消融实验（Ablation Study）

**目的**：量化每一步改造的独立贡献。

**参考集说明**：全部实验的 `ref CT` = 对应患者的真实 COPD CT（`copd_XXX_clean.nii.gz`，copd_024~029）。

| 步骤 | 操作描述 | 测试集 | 参考（ref） | 记录指标 |
|------|---------|:------:|:----------:|---------|
| **Exp-0** | 加载 `patchgan/best.pth`，原始 hardcoded target_stats（均值 -965，std 45） | copd_024~029 | 真实 COPD CT | ΔEI, PSNR, SSIM, HU KL, HU Mean/Std Error |
| **Exp-1** | 同上，但 target_stats 替换为每位患者的真实 c₃/c₄（零训练） | 同上 | 同上 | 同上 |
| **Exp-2** | 加载 `patchgan_cici/best.pth`，注入 5 维条件向量推理 | 同上 | 同上 | 同上 + 条件响应 r |

全部三步在同一验证集上运行，共享同一 `ref`，结果可直接比较：
- **Exp-1 − Exp-0** = 自适应 HU 校准的纯贡献（零训练）
- **Exp-2 − Exp-1** = 5 维条件向量 + FiLM 调制的纯贡献（2,434 参数）

#### 5.3.2 EI Sweep 实验（条件连续控制验证）

**协议**：

1. 选择一位验证集患者（如 copd_025），固定其 lesion mask
2. 构造 10 个条件向量，仅改变 c₁ (global_EI)：
   `c1 ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}`，
   其余 c₂-c₅ 保持该患者的真实值
3. 对每个条件向量运行推理，生成 10 个 fused CT
4. 测量每个生成结果的实际 EI (measured)
5. 绘制 `target_EI vs measured_EI` 散点图 + 线性回归

**预期结果**：

- 散点分布在 y = x 对角线附近
- Pearson r > 0.85（强正相关）
- 回归斜率 ∈ (0.7, 1.3)（近似线性响应）

**论文图表**：一张双坐标散点图，X 轴 = target EI，Y 轴 = measured EI，
附 r 值和回归方程。

#### 5.3.3 GOLD 分级分组实验

**协议**：

1. 按用户提供的 GOLD 分级将 29 例患者分为 4 组
2. 对每组运行条件化推理
3. 统计每组生成结果的 EI 分布
4. 进行组间统计检验

**统计方法**：

- 4 组比较：Kruskal-Wallis 检验（非参数，不要求正态分布，适合小样本）
- 两两比较：Dunn's post-hoc test with Bonferroni correction
- 显著性水平：α = 0.05

**预期结果**：

- GOLD Ⅳ 组的 EI_measured > GOLD Ⅰ 组（p < 0.05）
- 组间 EI 呈单调递增趋势

**论文图表**：四组箱线图（Box plot），附 p 值标注。

#### 5.3.4 特征贡献消融实验（5 维版本）

> 该实验验证影像学特征（c₁~c₄）与临床特征（c₅）各自的独立贡献。
> 注意：由于条件向量已精简为 5 维，消融设计也相应更新。

| 实验 | 条件向量构成 | 维度 | 目的 |
|------|:-----------|:----:|:-----|
| **Exp-A0** | 无条件（Baseline） | 0d | 基准线，对应 Exp-0 |
| **Exp-A1** | 仅 c₁~c₄（纯影像学） | 4d | 验证影像学特征的独立增益 |
| **Exp-A2** | 仅 c₅（纯 GOLD） | 1d | 验证临床分期特征的独立增益 |
| **Exp-A3** | 全部 c₁~c₅（完整 5 维） | 5d | 完整模型，对应 Exp-2 |

**预期排序**：Exp-A3 ≥ Exp-A1 > Exp-A2 > Exp-A0

**解读要点**：
- Exp-A1 与 Exp-A0 的差距 = 影像学特征的直接增益
- Exp-A3 与 Exp-A1 的差距 = GOLD 临床锚点的边际贡献
- Exp-A2 的单独测试揭示：仅凭粗粒度的 GOLD 分级（4 档），在没有连续影像学约束下，
  网络能否学到有意义的条件响应（预期很弱，但有学术展示价值）

---

## 6. 实施路线与代码改动清单

### 6.1 文件级改动总览

| 优先级 | 文件 | 类型 | 改动量 | 阶段 |
|:---:|------|:---:|:---:|:---:|
| 🥇 | `scripts/extract_patient_features.py` | 🆕 新建 | ~60 行 | ① |
| 🥇 | `inference_fuse.py` L246, L459-464 | ⚠️ 微调 | ~8 行 | ① |
| 🥇 | `run_phase3_pipeline.py` L1035, L1048 | ⚠️ 微调 | ~12 行 | ① |
| 🥈 | `src/.../conditioned_model.py` | 🆕 新建 | ~90 行 | ② |
| 🥈 | `dataset.py` L34, L235 | ⚠️ 微调 | ~8 行 | ② |
| 🥈 | `train.py` L145, L204 | ⚠️ 微调 | ~10 行 | ② |
| — | `data/clinical_data.csv` | 📋 用户提供 | 29 行 | ①② |
| **总计** | | | **~190 行** | |

### 6.2 数据依赖

```
用户需提供（精简后）:
  data/clinical_data.csv  (29 行: patient_id, GOLD)
  ← 注意: 原方案要求 4 列 (GOLD/FEV1_FVC/mMRC/CAT)，
    经特征审计后简化为仅需 GOLD 分级一列

自动生成:
  data/patient_features.json  (由 extract_patient_features.py 计算)
  → 包含每位患者的: global_EI, lesion_vol_ratio,
                    lesion_HU_mean, lesion_HU_std

已有（不变）:
  data/01_cleaned/copd_clean/copd_XXX_clean.nii.gz
  data/01_cleaned/copd_emphysema/copd_XXX_emphysema.nii.gz
  data/01_cleaned/copd_mask/copd_XXX_mask.nii.gz
  data/03_mapped/copd_XXX/copd_XXX_warped_lesion.nii.gz
  checkpoints/patchgan/best.pth  (22,587,123 params, epoch 148)
```

### 6.3 检查点兼容性保证

```
加载流程（阶段②，5 维条件向量版本）:

  1. backbone = InpaintingUNet()         ← 由 create_model('patchgan') 创建
     backbone.load_state_dict(ckpt['generator_state_dict'])
     ↓ 136 个 key 完美匹配 ✓

  2. model = ConditionedGenerator(backbone, cond_dim=5)
     ↓ model.state_dict() 的 key 格式:
       backbone.encoder.0.conv.conv.0.weight   ← 来自预训练，冻结
       backbone.encoder.0.conv.conv.0.bias     ← 来自预训练，冻结
       ...（136 个 backbone.* key）
       cond_encoder.0.weight  shape=(32, 5)    ← 新增，可训练 [5d输入]
       cond_encoder.0.bias    shape=(32,)      ← 新增，可训练
       cond_encoder.2.weight  shape=(64, 32)   ← 新增，可训练
       cond_encoder.2.bias    shape=(64,)      ← 新增，可训练
       film.gamma_proj.weight shape=(1, 64)    ← 新增，可训练
       film.gamma_proj.bias   shape=(1,)       ← 新增，可训练
       film.beta_proj.weight  shape=(1, 64)    ← 新增，可训练
       film.beta_proj.bias    shape=(1,)       ← 新增，可训练

  3. model.freeze_backbone()
     ↓ backbone.* 的 requires_grad = False
     ↓ 只有 8 个新增 key 可训练 (共 2,434 参数)

  关键验证:
     cond_encoder.0.weight 的 shape 从 (32,8) → (32,5)
     这是唯一受 cond_dim 影响的层，参数从 288 → 192
     其余所有层形状不变
```

### 6.4 回退安全保障

在任何时候，传递 `condition=None` 即可完全退化为原始行为：

- `ConditionedGenerator.forward(x, None)` → 等价于 `backbone.forward(x)`
- `fuse_lesion(..., patient_condition=None)` → 回退到 hardcoded target_stats

这意味着**所有现有实验结果可以 100% 复现**，新方案是纯增量式的。

---

*文档版本: v2.0 | 条件向量已从 8 维精简至 5 维 | 单模型聚焦（PatchGAN）| 冻结主干微调策略已确认*