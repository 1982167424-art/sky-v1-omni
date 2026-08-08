# 3D 模态：PointNet、NeRF、3DGS 与 Mesh 简介

## 一、3D 表示的四大主流形式

3D 世界在计算机中有多种表示方式，各有适用场景：

1. **点云（Point Cloud）**：N 个 (x, y, z) 坐标 + 可选法向量、颜色、语义标签。最原始，来自激光雷达（LiDAR）、深度相机、多视角立体匹配（MVS）。适合感知、检测、分割。
2. **体素（Voxel）**：空间 3D 网格，每个体素有占有/密度/颜色值。像 3D 版像素，规则但稀疏时空开销极大（32³ → 1024³ 涨 32k×）。高分辨率任务常用稀疏体素或哈希体素。
3. **网格（Mesh）**：顶点 V + 面 F（通常三角面）+ 可选材质贴图 (UV + Texture)。工业/游戏通用表示，可直接光栅化渲染，物理仿真友好。是最终交付格式。
4. **隐式场（Implicit Field / Neural Field）**：用神经网络 f(x, y, z) → (density, color/SDF) 连续描述整个 3D 空间，任意分辨率可微。代表作是 NeRF、3DGS 属于介于显式点与隐式神经之间的混合表示。

## 二、PointNet 与点云深度学习

点云是无序集合（置换不变）、每个点独立、数量不固定。CNN 需要规则网格，RNN 太慢。PointNet（Qi et al., CVPR 2017）首次解决了直接在原始点云上做深度学习的问题。

### 2.1 核心思想：对称函数 + 共享 MLP + 全局最大池化
对 N 个 D 维点云 P = {p₁, ..., p_N}, p_i ∈ R^D：
1. **逐点共享 MLP（MLP per-point, shared weights）**：每个点通过相同的 MLP 升维到 D'，保持置换不变；
2. **对称聚合**：用逐通道最大池化（max pooling）得到全局特征 g ∈ R^{D'}：
        g_j = max_{i=1..N} MLP(p_i)_j
   max 是对称函数（输入顺序打乱结果不变），从而整体保证置换不变性；
3. **T-Net 可学习对齐**：输入点云和中间层特征各加一个小网络 T-Net 预测 D×D 旋转矩阵，对点做姿态归一化，增强模型对旋转的鲁棒性。

### 2.2 PointNet 的两大任务头
- **分类头**：全局特征 g → MLP → K 类分类损失；
- **分割头**：全局特征 g 与每个点的局部特征 concat → 逐点 MLP → 每个点的语义标签。

### 2.3 PointNet++：多尺度分组（MSG + MRG）
PointNet 缺少像 CNN 那样的多尺度局部感受野，对局部几何结构捕捉不足。PointNet++ 提出分组下采样层级结构：
1. **FPS 最远点采样**：从 N 点采 N₁ < N 个"代表性"中心点；
2. **Ball Query**：在每个中心点半径 r 内取 K 个邻域点构成局部 Group；
3. **Set Abstraction**：每个 Group 独立做一次 mini-PointNet，得到新的 N₁ × D' 特征；
4. **MSG / MRG**：多尺度分组（不同 r、不同 K 并行）或多分辨率分组（不同 SA 层组合），提升对密度不均匀点云的鲁棒性。

PointNet 系列在自动驾驶 LiDAR 3D 检测、工业零件质检、AR 空间点云语义分割上仍是基础骨干。

## 三、NeRF：神经辐射场与新视角合成

NeRF（Mildenhall et al., ECCV 2020 Best Paper Honorable Mention）是 3D 视觉过去 5 年最有影响力的工作，首次用一个 MLP 隐式表示整个场景的颜色 + 密度，通过体渲染可微，端到端从多视角图片反演 3D 几何。

### 3.1 神经辐射场 f_θ
MLP 输入：一条光线上采样点的 (x, y, z) + 观察方向 (d_x, d_y, d_z)；
输出：该点的体密度 σ（标量）+ 该点在方向 d 下的发射颜色 c = (r, g, b)。
为了表示高频细节，(x, d) 先做 **位置编码（Positional Encoding / PE）**：将每个坐标映射到 sin/cos 高频基：

    γ(x) = [sin(2⁰πx), cos(2⁰πx), ..., sin(2^{L-1}πx), cos(2^{L-1}πx)]

L 通常取 10（坐标）/4（方向），把坐标从 3 维升到 60 维 / 24 维，使 MLP 能表达照片级细节。

### 3.2 可微体渲染（Volume Rendering Equation）
对相机发出的像素射线 r(t) = o + t·d，沿 t ∈ [t_n, t_f] 采样 N 个点 {t_i}，计算累计颜色：

    C(r) = Σ_i T_i (1 - exp(-σ_i δ_i)) c_i,  T_i = exp(-Σ_{j=1}^{i-1} σ_j δ_j)

其中 T_i 是射线穿透到第 i 个点的透射率（透明度衰减），δ_i = t_{i+1} - t_i 是相邻采样点间距。整个过程完全可微，可直接用渲染图与真实图的 L2 / LPIPS loss 端到端训练 MLP f_θ。

### 3.3 NeRF 关键加速与扩展
- **Mip-NeRF**：用圆锥采样（I/Gaussian）替代针孔采样，消除锯齿；
- **NeRF++ / Instant NGP**：哈希编码 + 小 MLP，训练从天级降到分钟级；
- **360° / unbounded NeRF**：用反演球面处理室外大场景；
- **NeuS / VolSDF**：把 σ 替换为 Signed Distance Function (SDF)，训练出干净 Mesh；
- **Human NeRF / AniNeRF**：动态人体、动态物体；
- **Nerfstudio**：开源一体化 NeRF 框架。

## 四、3DGS：3D 高斯泼溅——实时高质量新视角渲染

NeRF 虽画质惊艳但渲染慢（1 张图数秒），难用于实时交互。3DGS（3D Gaussian Splatting, Kerbl et al., SIGGRAPH 2023 Best Paper）把 NeRF 的神经隐式场替换为"显式的、带协方差的、带球谐颜色的 3D 高斯团集合"，实现了 **训练 30~60 分钟 + 1080p 100+ FPS 实时渲染 + 画质超 NeRF** 的惊人效果，是目前 3D 重建/数字人/AR 最火热的技术。

### 4.1 表示：N 个 3D 各向异性高斯
场景由 N 个 3D Gaussian 组成，每个高斯参数化：
- **位置** μ ∈ R³；
- **协方差** Σ ∈ R^{3×3}，表示为四元数旋转 q + 3 维缩放 s → Σ = R S S^T R^T（保证 PSD）；
- **不透明度** α ∈ [0,1]；
- **球谐颜色** SH = {c_l^m}_{l=0..L-1, m=-l..l}，L=1~4 阶，球谐系数按观察方向计算出最终 RGB（比 NeRF 的 view-dependent color 更紧凑）。

### 4.2 可微光栅化（Differentiable Rasterization）
渲染把每个 3D 高斯投影到 2D 成像平面，得到 2D 椭圆高斯：

    μ_2D, Σ_2D = Project_World_To_Image(μ, Σ, K, T_cw)

然后按深度从小到大对每个像素沿椭圆做 **Alpha Blending 叠加**（相当于 NeRF 体渲染的离散 2D 版本）：像素值 = Σ_i α_î c_i (1 - α_î_pre)。整个光栅化用 CUDA Tile-based 实现，前向/反向完全可微。

### 4.3 自适应密度控制（Density Control）
初始化通常来自 SfM（COLMAP）重建的稀疏点云。训练中：
- **分裂（Split）**：大且高透明度的高斯（表示区域欠重建）→ 一分为二两个更小的高斯；
- **克隆（Clone）**：渲染误差大的地方高斯数量不够 → 复制附近高斯；
- **剪枝（Prune）**：不透明度极低的高斯 → 直接删除。

这种分裂/克隆/剪枝每 100~1000 步执行一次，让高斯数量与分布自适应场景复杂度：小物体 50k 高斯，大房间 200~500k 高斯，大场景城市场景 5M+。

### 4.4 3DGS 发展与与 Mesh 的桥接
3DGS 后续有：4DGS（动态时空高斯）、ScaffoldGS（加 Scaffold 保留 Mesh 拓扑）、GaussianShader（可编辑材质）、LaserGS（激光雷达融合）。将 3DGS 转为高质量 Mesh 的常用流水线是 Poisson Surface Reconstruction / Neural Surface Extraction（如 NeuS），用 GS 的密度/法线监督学习 SDF，最后 Marching Cubes 抽 Mesh + 烘焙纹理贴图。

## 五、Mesh：工业/游戏通用 3D 格式与生成

Mesh（三角形网格）是最经典也最工程实用的 3D 表示，由三要素构成：
- **顶点 V ∈ R^{|V|×3}**：空间坐标；
- **面 F ∈ {0..|V|-1}^{|F|×3}**：每个三角面的三个顶点索引；
- **属性**：法向量 n ∈ R^{|V|×3}、UV 坐标 uv ∈ R^{|V|×2}、材质贴图 texture.png、材质 BSDF 参数（PBR 的金属度、粗糙度、法线贴图、AO 贴图等）。

### 5.1 Mesh 生成的三大范式
1. **模板变形（Template-based）**：给定 SMPL / SMPLX 人体模板、Sphere/Cube 基元模板，训练网络预测每个顶点位移 ΔV，天然保证流形 watertight，适合人体、脸、标准化物体；
2. **Marching Cubes from Fields**：先训练 SDF/NeRF/3DGS 的隐式场或密度场，再用 MC 算法在固定分辨率网格上提取等值面，最后减面 + 重拓扑 + UV 展平 + 贴图烘焙。通用性最强，是 GET3D、InstantMesh、DreamFusion 的主流；
3. **直接自回归生成（Mesh Transformer）**：将 Mesh 序列化为面序列（Faceformer）或 Token，训练 LLM 风格 Transformer 直接生成，如 MeshGPT、Shape-e。

### 5.2 Mesh 质量指标
工业上关心：是否流形（Manifold）、是否封闭（Watertight，保证可 3D 打印/物理仿真）、面数合理、法向量一致朝外、无自相交、UV 无重叠、贴图分辨率足够。常用工具：Blender（手工编辑/重拓扑）、Meshlab（网格处理/简化）、Instant Meshes（自动重拓扑）、xatlas（UV 展开）、RizomUV（专业 UV）。

## 六、3D 模态与 sky-v1-omni 五模态融合

在 sky-v1-omni 的统一框架中，3D 模态承担"理解真实世界几何"与"输出可生产 3D 资产"两大任务：
- **3D 理解头**：PointNet++ 风格的点云编码器 + 跨模态对齐投影，把用户上传的点云/GLB 模型编码为 token，LLM 可回答 "这个零件哪里有缺陷？"；
- **3D 生成头**：Text → SDF / 3DGS / Mesh，配合图像模态生成的多视角参考图（Zero-1-to-3 / SyncDreamer）和视频模态的 360° 旋转展示视频，一次交付 Text → Image → Video → 3D 完整创作链路；
- **3D + 视频 + 音频 联动**：给定 3D 场景 + 对话剧本，自动摆机位（相机路径）→ 渲染视频（3DGS / Rasterizer）→ TTS 配音 → 字幕，生成产品级宣传视频。这正是多模态统一模型相对于分散的"单模型堆叠"的真正壁垒所在。
