# 摘要

慢性阻塞性肺疾病（COPD）是全球第三大致死疾病，病理改变在肺内呈高度空间异质性分布。传统二维CT阅片难以全面展示病灶的三维空间分布特征，制约了临床对疾病严重程度和空间格局的精准判读。数字孪生技术可为构建物理器官的虚拟映射体开辟新路径，但面向COPD肺器官的数字孪生构建与三维交互可视化方法尚处于早期探索阶段。本文针对"基于CT影像的肺器官数字孪生可视化方法"这一问题，提出了从原始CT影像到三维可交互数字孪生肺的完整技术链路。

首先，提出"健康底座+病灶叠加"的数字孪生肺构建范式。从389例临床记录中筛选37例健康受试者的双相CT数据，以SyN对称微分同胚配准为核心算法，经5轮迭代群组平均融合为标准肺CT模板及六类解剖标签底座。质量验证显示底座的平均Dice系数为0.842±0.031，Wasserstein距离为32.20±20.41 HU，变形场折叠率为0.000%。

其次，设计了融合HU物理约束的PatchGAN病灶纹理生成方法。以LAA-950阈值从29例COPD患者CT中提取肺气肿掩膜，经SyN配准映射至标准空间后，利用PatchGAN深度图像修复网络在底座上合成COPD病理纹理。四分量组合损失函数把-950 HU临床阈值编码为优化目标，区域自适应直方图校准以Z-score线性变换结合Gamma压暗修正消除推理阶段的密度偏差，LAA-950保真度偏差控制在2个百分点以内，清晰度较直接配准方法提升约546.1%。

最后，搭建了包含十四个功能模块的COPD数字孪生肺三维可视化系统。系统基于四层分离架构和纯Python全栈技术实现Web端三维交互，以四类颜色语义正交的色彩映射方案把多维病理指标编码为三维空间的直觉视觉信号，五维度加权评分算法把分析结果转化为临床可操作的量化风险评级。

**关键词：** 数字孪生；慢性阻塞性肺疾病；生成对抗网络；直方图匹配；三维交互可视化

---

# Abstract

Chronic obstructive pulmonary disease (COPD) is the third leading cause of death worldwide, characterized by spatially heterogeneous pathological changes. Conventional two-dimensional CT interpretation cannot fully capture the three-dimensional distribution of lesions, limiting assessment of disease severity and spatial patterns. This thesis proposes a pipeline from raw CT images to an interactive three-dimensional digital twin lung for COPD visualization.

First, a "healthy base plus lesion overlay" paradigm is proposed. CT data from 37 healthy subjects, screened from 389 clinical records, are fused into a standard lung CT template through five iterations of SyN symmetric diffeomorphic group registration, achieving a mean Dice coefficient of 0.842±0.031, a Wasserstein distance of 32.20±20.41 HU, and a deformation field folding rate of 0.000%.

Second, a PatchGAN-based lesion texture generation method with Hounsfield unit constraints is designed. Emphysema masks from 29 COPD patients extracted via the LAA-950 threshold are mapped to the template space through SyN registration. A four-component loss function encodes the -950 HU clinical threshold as an optimization target. Region-adaptive histogram calibration combining Z-score transformation and Gamma correction eliminates residual density deviations, controlling LAA-950 fidelity deviation within two percentage points and improving sharpness by 546.1% over direct registration.

Third, a three-dimensional visualization system comprising fourteen functional modules is developed. Built on a four-layer architecture with a Python Web frontend, the system employs four chromatically orthogonal color mapping schemes to encode pathological indices into three-dimensional visual signals. A five-dimension weighted scoring algorithm converts results into clinical risk ratings.

**Keywords:** digital twin; chronic obstructive pulmonary disease; generative adversarial network; histogram matching; three-dimensional interactive visualization

