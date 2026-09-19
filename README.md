# Efficient Online Open-Vocabulary 3D Scene Perception for Edge Robotics

面向端侧机器人的**在线开放词汇 3D 场景感知** —— 从 RGB-D 流实时构建可文本查询的 3D 实例地图，
并把点云骨干压缩部署到国产 NPU（RK3588）。

**English** — An online, training-free open-vocabulary 3D instance perception pipeline for edge robots:
RGB-D stream → 2D open-vocabulary detection & segmentation → 3D lifting → incremental voxel mapping →
geometric instance association & tracking → text-queryable instance map. The point backbone is then
rewritten into an NPU-deployable form and shipped to a Rockchip RK3588 (6 TOPS INT8).

---

## 目录

- [效果](#效果)
- [这是什么](#这是什么)
- [系统架构](#系统架构)
- [量化结果](#量化结果)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [技术难点与解法](#技术难点与解法)
- [局限](#局限)
- [参考](#参考)

---

## 效果

<!-- 放一张 tracking_contact_sheet.png 或 3D 实例图 -->

![3D instance tracking](docs/assets/tracking_contact_sheet.png)

在 Replica `office0` 的 61 帧（0–600，步长 10）上，与 GT 实例掩码对比：

| 指标 | 数值 |
|---|---|
| 实例 AP@IoU 0.25 | **0.703** |
| 实例 AP@IoU 0.50 | 0.314 |
| 语义标签一致性 | **0.975** |
| GT 对象单轨保持率 | 0.556 |

---

## 这是什么

一条**免训练**的在线 3D 感知流水线，输入是带位姿的 RGB-D 序列，输出是一张
**类别无关的 3D 实例地图**，每个实例带一个**开放词汇语义嵌入**（CLIP 空间），
因此可以用任意自然语言查询场景（"可以坐的东西在哪"、"哪里有插座"）。

设计上的两个关键取舍：

1. **实例形成不依赖语义。** 物体的"是一个独立个体"可以纯靠几何与多帧一致性判定，
   语义只在实例稳定后才贴上去。这样实例分组不会因为见到没见过的类别而崩。
2. **不用稀疏卷积。** 点式方案零额外依赖，天然处理在线融合点云的变密度问题，
   而且**在 NPU 上可量化、可部署** —— 稀疏卷积的动态索引在 RKNN 上几乎不可实现。

---

## 系统架构

```mermaid
flowchart TD
    A["RGB-D 流 + 位姿"] --> B["Grounding DINO<br/>开放词汇 2D 检测"]
    B --> C["SAM 2<br/>实例分割"]
    C --> D["背景块过滤<br/>按框面积占比"]
    D --> E["深度反投影<br/>2D mask → 3D 点集"]
    E --> F["体素建图<br/>增量融合"]
    F --> G["几何/表面实例关联<br/>+ 在线跟踪"]
    G --> H["3D 实例地图"]
    H --> I["CLIP 语义嵌入<br/>文本可查询"]

    E --> J["Stage 5 点式骨干<br/>PointEncoder"]
    J --> I
    J --> K["RKNN 导出<br/>→ RK3588 NPU"]

    style K fill:#085041,stroke:#5DCAA5,color:#9FE1CB
    style J fill:#085041,stroke:#5DCAA5,color:#9FE1CB
```

---

## 量化结果

> 每个数字的来源文件与复现命令见 **[`docs/RESULTS.md`](docs/RESULTS.md)**；
> 指标定义（AP / 碎片率 / 四个 IoU 口径 / 延迟口径）见 **[`docs/METRICS.md`](docs/METRICS.md)**。

### 1. 在线感知（Replica office0，61 帧）

> office0 是服务器上**唯一**有 RGB-D + 位姿序列的场景，所以这一项是单场景结果。

| 指标 | 数值 |
|---|---|
| AP@IoU 0.25 | 0.703 |
| AP@IoU 0.50 | 0.314 |
| 语义标签一致性 | 0.975 |
| 关联：匹配到的 GT 对象 | 9 |
| 关联：预测轨道数 | 13 |

### 1b. 单进程推理延迟（RTX 3090，61 帧）

原来的 30.74 s/帧里有 **90% 是脚手架开销**（每帧 spawn 子进程、重复加载权重、写盘、渲染预览）。
改成单进程、模型只加载一次、默认不落盘后：

| 阶段 | 均值 | 中位数 | p95 |
|---|---|---|---|
| 读帧 | 30.5 ms | 28.7 ms | 40.9 ms |
| Grounding DINO（2D 检测） | 207.2 ms | 203.5 ms | 247.9 ms |
| SAM 2（实例分割） | 92.0 ms | 88.1 ms | 130.3 ms |
| 深度反投影 | 58.6 ms | 52.2 ms | 104.7 ms |
| **合计** | **388.4 ms** | **392.6 ms** | 471.3 ms |

**2.57 FPS，比旧口径快 79×。** 纯 2D 推理占 77.0%。
再加上写盘与预览渲染会变成 2078.4 ms/帧 —— 证明剩余开销**全在可视化，不在推理**。

复现：`python -m tools.benchmark_latency --frames 0 10 20 ... --warmup 3`

### 2. Stage 5 点式骨干（87 类，975,975 参数）

**按场景划分，不是按块划分** —— 训练/验证/测试用的是**完全不同的房间**：

| 划分 | 场景 | 点数 | 准确率 | mIoU | 宏平均 mIoU | 支撑度≥200 的 mIoU | 频次加权 IoU |
|---|---|---|---|---|---|---|---|
| train | 12 个（apartment / frl_apartment / hotel / office） | 3.6M | — | — | — | — | — |
| val | office_2 / office_3 / office_4 | 591,964 | 0.540 | 0.152 | 0.198 | 0.186 (22/30 类) | 0.477 |
| **test** | room_0 / room_1 / room_2 | 593,653 | **0.587** | **0.163** | **0.225** | **0.193 (32/38 类)** | 0.473 |

逐场景：

| 场景 | 准确率 | mIoU | 支撑度≥200 的 mIoU |
|---|---|---|---|
| office_2 | 0.596 | 0.238 | 0.336 |
| office_3 | 0.528 | 0.147 | 0.212 |
| office_4 | 0.496 | 0.210 | 0.260 |
| room_0 | 0.653 | 0.245 | 0.286 |
| room_1 | 0.558 | 0.217 | 0.241 |
| room_2 | 0.550 | 0.213 | 0.247 |

**两个值得注意的点**：

1. **测试集（从未参与选模型）比验证集更好**（mIoU 0.163 vs 0.152，宏平均 0.225 vs 0.198）
   —— 没有在验证集上过拟合。测试集里房间类场景见到的类别还更多（38 vs 30 类）。
2. 表现好的类别是**结构面 + 大件家具**：ceiling 0.90 / rug 0.56 / lamp 0.55 /
   floor 0.54 / wall 0.49 / table 0.47 / chair 0.38。差的都是长尾小物体
   —— 这也是宏平均 mIoU 被拖低的原因，所以同时报告"支撑度≥200"与"频次加权"两个口径。
   详见 `docs/METRICS.md`。

> 数据来源：`outputs/stage5/run_s5b_metricfix/{eval_val,eval_test}.json`。
> 复现：`python -m tools.evaluate_point_encoder --checkpoint .../best.pt --split test`

### 3. NPU 部署可行性（ONNX 静态分析，RKNN 支持表比对）

| 档位 | 导出器 | 总节点 | 可上 NPU | 必须留 CPU |
|---|---|---|---|---|
| 完整图（含 Morton 编码 + kNN） | TorchScript opset 12 | **导出失败** | — | — |
| 完整图（仅作量级参考） | dynamo opset 17 | 3425 | 2662 (77.7%) | **678 (19.8%)** |
| **Morton 编码移出** | TorchScript opset 12 | 719 | 665 (**92.5%**) | **24 (3.3%)** |
| + kNN 移出 | TorchScript opset 12 | 639 | 594 (93.0%) | 15 (2.3%) |

含 Morton 编码的完整图**在部署路径下根本导不出来**，而且不是 opset 的问题：

| opset | 结果 |
|---|---|
| 12 / 13 / 15 / 17 | `ONNX export does NOT support exporting bitwise AND` |
| 18 / 19 / 20 | `ONNX export does NOT support exporting bitwise OR` |

也就是说，Morton 的位运算**连 ONNX 这一关都过不去**，不需要等到上 NPU 才被拒绝。
只把 Morton 编码移出图，CPU 侧负担就从 678 个节点降到 24 个（同一导出器口径下 −96%）。

### 4. RKNN 可部署形态（改写后）

| 指标 | 数值 |
|---|---|
| 部署图节点数 | 705 |
| 算子种类 | 22 |
| **不支持的算子** | **0**（Erf / Sqrt / ReduceL2 全部消灭） |
| 数值等价：最大相对偏差 | **5.5e-4** |
| 数值等价：语义 argmax 一致率 | **99.951%** |

---

## 快速开始

```bash
# 环境：Python 3.10+，PyTorch 2.x（开发环境是 torch 2.5.1+cu121）
pip install -r requirements.txt

# 0. 准备数据：把 Replica 原始场景放成 datasets/raw/replica_v1/<scene>/
#    再把带 RGB-D + 位姿的序列放成 datasets/processed/Replica/<scene>/
#    （results/frame*.jpg、results/depth*.png、traj.txt）

# 1. 生成 GT 实例掩码（从 Replica mesh_semantic.ply 反投影）
python -m tools.generate_gt_instance_masks \
    --scene-directory datasets/processed/Replica/office0 \
    --replica-scene office_0 \
    --output-directory outputs/gt/office0

# 2. 跑在线流水线（回放采样帧）
python -m tools.run_instance_sequence \
    --scene-directory datasets/processed/Replica/office0 \
    --run-directory outputs/experiments/office0_run

# 3. 与 GT 对比评测（AP + 轨道碎片率/纯度）
python -m tools.evaluate_against_gt \
    --tracking-json outputs/experiments/office0_run/association/tracking.json \
    --segmentation-root outputs/experiments/office0_run/segmentation \
    --gt-root outputs/gt/office0 \
    --output outputs/experiments/office0_run/gt_eval.json

# 4. 构建点云数据集并训练 Stage 5 点式骨干（自动 12/3/3 按场景划分）
python -m tools.build_mesh_point_dataset --output-directory outputs/point_dataset
python -m tools.train_point_encoder \
    --dataset-root outputs/point_dataset \
    --output-directory outputs/stage5/run \
    --epochs 60 --num-points 8192 --lr 2e-3

# 5. 跨场景评测（val / test 都是训练时没见过的房间）
python -m tools.evaluate_point_encoder \
    --checkpoint outputs/stage5/run/best.pt \
    --dataset-root outputs/point_dataset --split test \
    --output outputs/stage5/run/eval_test.json

# 6. 导出 RKNN 可部署形态（含数值等价性校验）
python -m tools.rknn_export --onnx-out deploy/point_encoder_rknn.onnx
```

### 测试与审计

```bash
python -m tools._smoke_test_rknn_export   # RKNN 导出路径：4 条工程保证
python -m tools.rknn_op_audit             # 算子兼容性审计（不需要任何硬件）
python -m tools.benchmark_latency --frames 0 10 20 30 --warmup 3   # 单进程延迟基准
```

`rknn_op_audit` 与 `benchmark_latency` 都**不需要 RK3588 硬件**，
前者做 ONNX 静态分析，后者用桌面 GPU 测推理延迟。

---

## 目录结构

```
.
├── src/
│   ├── perception/      # 2D 检测/分割、CLIP 编码
│   ├── geometry/        # 反投影、体素建图、几何关联
│   ├── mapping/         # 增量实例地图与跟踪
│   ├── models/          # PointEncoder、RKNN 导出
│   └── datasets/        # Replica 读取、点云块数据集
├── tools/               # 所有可执行脚本（python -m tools.xxx）
├── docs/                # 指标口径、审计报告、设计文档
└── outputs/             # 实验产物（gitignore）
```

---

## 技术难点与解法

这一节记录踩过的坑和对应的解法 —— 也是这个项目里最花时间的部分。

### 1. 推理时漏了归一化，整场景评测直接腰斩

**现象**：按块评测 mIoU 0.158，整场景评测只有 0.066。

**根因**：训练时每个点云块都会被归一化（平移到质心 + 缩放到单位球），
但推理时 `encode_cloud_in_blocks` 直接把**世界坐标**喂给了模型。
第一层 MLP 和邻域相对坐标全部落在训练分布之外。

**定位方式**：把同一片点云整体 ×3 并平移 (100, −50, 20)，
比较归一化前后的预测一致性 —— 归一化后 **1.0000**，不归一化只有 **0.3580**。

**修复后**：准确率 0.369 → 0.523，宏平均 mIoU 0.066 → 0.142，
30 个类别里有 IoU 的从 11 个升到 21 个。

### 2. 训练日志打印的是最弱的那条分支

**现象**：验证 mIoU 曲线一直平在 0.12–0.16，但监督语义损失从 0.64 掉到 0.14。

**根因**：开了文本嵌入之后，`evaluate()` 用**开放词汇分支**
（`text_embedding @ text_bank.T()`）做预测，而这条分支的损失权重只有 0.1。
`best.pt` 也就一直按这个最弱的分支在选。

**修复**：同时评测两条分支，`SELECTION_METRIC = "semantic.miou"`。
修复后两种准则选出的 checkpoint 确实不同（epoch 49 vs 34）。

### 3. 宏平均 mIoU 被长尾噪声主导

87 个类别里大部分在验证集上只有个位数点数，IoU 恒为 0，把宏平均拖到 0.167，
而频次加权是 0.454。**修复**：从同一个混淆矩阵同时产出
`miou` / `miou_supported`（≥200 点）/ `frequency_weighted_iou` 三个口径 + 每类支持度。

> **注意区分**：支持度**大**但 IoU 恰好为 0 的类别是**真失败**，不是长尾。
> 当前仍有 7 个这样的类别（bench / blinds / non-plane / pipe / stool / tv-stand / wall-plug）。

### 4. 一个 3.3 GB 的临时张量把评测卡死

`((block[:,None,:] - ref[None,:,:])**2).sum(-1)` 在 B=1024 / N=50000 / C=16 时
会实体化 3.3 GB 的中间张量。改成展开形式
（`b² − 2b·rᵀ + r²`）后峰值降到 164 MB，整场景评测 **540 s → 102 s（5.3×）**。

### 5. 把骨干改成 NPU 能跑的样子

RK3588 的 NPU 走 RKNN 工具链，算子覆盖比 TensorRT 差得多。静态分析图节点后发现，
含 Morton 编码的完整图**在部署路径下根本导不出来**：

| opset | 部署路径（TorchScript）的结果 |
|---|---|
| 12 / 13 / 15 / 17 | `ONNX export does NOT support exporting bitwise AND` |
| 18 / 19 / 20 | `ONNX export does NOT support exporting bitwise OR` |

**每一个 opset 都失败。** 这比"RKNN 不支持位运算"更强 ——
Morton 的位运算连 ONNX 这一关都过不去，不需要等到上 NPU 才被拒绝。

把动态算子移出图之后，剩下的障碍是三类算子：

- **Morton 编码（位运算）** 是位运算，RKNN 完全不支持；
- **kNN（TopK / Gather / GatherElements）** 是动态索引，NPU 张量引擎不支持非连续内存访问；
- **GELU 的 `Erf`**、**LayerNorm / L2 归一化的 `Sqrt` / `ReduceL2`** 不在支持列表里。

**解法**：

| 问题 | 解法 | 结果 |
|---|---|---|
| Morton 编码在图内 | 索引提到图外，由 CPU 提供 | CPU 侧负担 **678 → 24 节点**（−96%） |
| kNN 动态索引 | 留 CPU（本来就边界清晰） | 只剩 `GatherElements` ×15 |
| `Erf`（GELU） | 换 tanh 近似 | 偏差 4.7e-4 |
| `Sqrt`（LayerNorm） | 换 `Pow(var+eps, -0.5)` | 偏差 4.8e-7 |
| `ReduceL2`（L2 归一化） | 换 `Pow(Σx², -0.5)` | 偏差 6.0e-8 |
| `Sqrt`（反距离加权） | 换 `Pow(‖Δ‖²+ε², -0.5)` | 偏差 0 |

**训练路径完全不动** —— 替换发生在深拷贝出的副本上，`state_dict` 键不变，
已有 checkpoint 双向兼容。整模型数值等价由 `_smoke_test_rknn_export.py` 把关。

最终部署图 **705 节点 / 22 类算子 / 0 个不支持的算子**。

> 一个值得记录的诊断：`Pow(x, -0.5)` 会不会被导出器拆成 `Sqrt + Reciprocal`？
> **取决于指数是 Python float 还是张量**。实测 `torch.pow(v, -0.5)`（float 标量）
> 得到干净的 `Pow`；而 `torch.rsqrt(v)` 变成 `Sqrt + Div`、
> `1.0/torch.sqrt(v)` 变成 `Sqrt + Reciprocal`，两个都白改。

> 一个意外收获：我们的 Morton 序子采样天然满足 RK3588 的内存局部性要求
> （公开的 RK3588 PointNet 部署记录里，作者必须手动按空间排序才能让 NPU
> 沿点维规约跑起来）。这本来是为了避免 FPS 的 O(N·M) 代价做的选择，
> 在 NPU 部署上变成了优势。

---

## 局限

诚实列出，避免误导：

- **在线流水线只评了单场景。** 完整评测只在 Replica `office0` 上做了。
  原因是**服务器上只有 office0 有 RGB-D + 位姿序列** —— 其余 17 个场景的
  原始数据只有 mesh / semantic，没有帧。Stage 5（点式骨干）不受这个限制，
  已经做了 12/3/3 的跨场景划分（见 §2）。
- **精度不占优。** AP@0.50 只有 0.314，掩码 IoU ≥0.5 的比例约 63.5%。
  过分割是主要问题（预测轨道数 13 vs GT 对象 9），根因在 2D 分割前端 ——
  这与 OVI-MAP 论文 Table 5 的结论一致（SAM2 → CropFormer 让 instance AP50 从 27.8 涨到 50.8）。
- **开放词汇还很弱。** 开放词汇分支 mIoU 只有 0.127，离可用还有距离。
- **RK3588 尚未实测。** 部署可行性来自 ONNX 静态分析 + RKNN 支持表比对，
  **没有在真机上跑过**。RKNN 支持表依据的是 toolkit v1.7.5 文档，
  早于 RK3588 的 toolkit2 线，逐算子细节仍需用
  `RKNN_OP_Support_And_Limit.xlsx` 复核。
- **功耗未测。** 需要 USB-C 功率计。
- **端到端延迟不是实时的。** 单进程基准 388.4 ms/帧（2.57 FPS，RTX 3090），
  其中 2D 检测+分割占 77%。这是**桌面 GPU 上的推理延迟**，
  不等于 RK3588 上的表现，两者之间还隔着量化与 CPU/NPU 切分。

---

## 参考

本项目在实现过程中参考并对比了以下工作：

- **ESAM** — *EmbodiedSAM: Online Segment Any 3D Thing in Real Time* (ICLR 2025)
- **ESAM++** — SFPN: 用稀疏特征金字塔替换 3D sparse UNet
- **OVI-MAP** — *Open-Vocabulary Instance-Semantic Mapping* (CVPR 2026)
- **FOLK** — *Fast Open-Vocabulary 3D Instance Segmentation via Label-guided Knowledge Distillation*
- **AutoSeg3D** — *Online Segment Any 3D Thing as Instance Tracking* (NeurIPS 2025)
- **OpenMask3D** — *Open-Vocabulary 3D Instance Segmentation*
- **PointNet++** — Set Abstraction / Feature Propagation 结构
- **Grounding DINO** / **SAM 2** — 开放词汇检测与分割
- **Replica** — 数据集

---

## 许可

MIT
