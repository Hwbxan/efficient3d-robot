# 指标口径

这个项目同时有**两套完全不同的评测**：2D/跟踪那套（对着 GT 实例掩码算）和
Stage 5 语义分割那套（对着 GT 类别算）。两套指标不能混着引用。
本文把每个数字的来源、定义和"多少算好"写清楚。

---

## 一、在线感知：检测与关联（`tools/evaluate_against_gt.py`）

输入是跟踪结果（`tracking.json`）、逐帧分割结果（`segmentation/`）和 GT
（`outputs/gt/<scene>/`，由 `tools/generate_gt_instance_masks.py` 从
`mesh_semantic.ply` 反投影生成）。输出在 `gt_eval*.json`。

### 1.1 检测（逐帧，2D）

对每一帧，把预测掩码与 GT 实例掩码按 IoU 匹配，然后按帧聚合。

| 字段 | 含义 |
|---|---|
| `ap` | 该 IoU 阈值下的 Average Precision |
| `precision` / `recall` | 该阈值下的精确率 / 召回率 |
| `tp` | 真阳性数 |
| `predictions` | 预测实例总数（所有帧累加） |
| `gt_instances` | GT 实例总数（所有帧累加；已按 `--min-gt-area` 与类别过滤） |

**注意分母**：`gt_instances` 是"实例 × 帧"，不是场景里的物体数。
office0 的 189 = 61 帧 × 平均约 3 个达标 GT 物体。

### 1.2 关联（3D 跟踪）

把 GT 物体和预测轨道做二分匹配（阈值 `--association-iou`，默认 0.5）。

| 字段 | 含义 | 方向 |
|---|---|---|
| `matched_gt_objects` | 被至少一条轨道覆盖的 GT 物体数 | — |
| `matched_tracks` | 参与匹配的轨道数 | — |
| `single_track_ratio` | 匹配上的 GT 物体里，**只由一条轨道覆盖**的比例 | ↑ 越高越好 |
| `pure_track_ratio` | 轨道里**只覆盖一个 GT 物体**的比例 | ↑ 越高越好 |
| `fragmentation_mean` | 每个匹配上的 GT 物体平均被拆成几条轨道 | ↓ 越低越好（1.0 是完美） |
| `fragmented_gt_objects` | 具体哪些物体被拆开了，以及拆成了哪几条轨道 | 诊断用 |
| `impure_tracks` | 具体哪些轨道混进了多个 GT 物体 | 诊断用 |

**过分割的直接读数**：`matched_tracks` ÷ `matched_gt_objects`。
office0 上是 13 ÷ 9 = 1.44，即平均每个物体被拆成 1.44 条轨道 ——
这正是"过分割是主要误差来源"的量化依据。

### 1.3 语义标签一致性

| 字段 | 含义 |
|---|---|
| `consistent` / `total` | 关联成功的观测里，预测标签与 GT 标签一致的数量 |
| `ratio` | 上者之比 |

注意这是**在已关联的观测上**算的，不包含漏检 —— 别把它当检测精度用。

### 1.4 参数扫描

`score_sweep` 与 `area_sweep` 是对 `--min-score` 和 `--max-area-fraction`
（背景块过滤阈值）做的扫描。主表默认取 `max_area_fraction = 0.4`。

---

## 二、Stage 5 点式骨干：语义分割（`tools/evaluate_point_encoder.py`）

输入是点云（网格采样或在线融合），输出逐点类别。指标在**忽略 class = -1**
（无标注点）之后统计。

### 2.1 四个 IoU 口径，各有各的用途

| 字段 | 定义 | 为什么要它 |
|---|---|---|
| `accuracy` | 逐点准确率 | 最直观，但**会被大类主导**（floor/wall/ceiling 占绝大多数点） |
| `miou` | 对"在点云里出现过的类"求 IoU 平均 | 类别级平均；但仍按合并点云统计，大场景权重更大 |
| `macro_miou` | **每个场景各算一个 `miou`，再对场景取平均** | 每个场景等权，避免大场景吃掉小场景 |
| `miou_supported` | 只在 **GT 点数 ≥ `min_support`（默认 200）** 的类上求平均 | 剔掉"只有几十个点"的长尾类，反映**可用**精度 |
| `frequency_weighted_iou` | Σ(类点数 × 类 IoU) ÷ Σ 类点数 | 按实际点分布加权，接近"随机取一个点判对"的概率 |

**`miou` 与 `macro_miou` 的区别容易搞混**：前者在合并后的混淆矩阵上按类平均，
后者先按场景算再平均。两者都报，是为了避免"用一个数字讲故事"。

### 2.2 为什么必须报 `miou_supported`

Replica 的 87 类里，很多类在单个场景只有几十个点（比如 `coaster`、`candle`）。
这些类的 IoU 抖动极大：预测对一个点，IoU 就能从 0 跳到 0.5；预测错一个点，
就从 0.5 掉到 0。它们把宏平均 mIoU 拉低，但**不代表模型在可用类别上不行**。

所以本项目一律同时报 `miou`、`miou_supported`、`frequency_weighted_iou` 三个数，
并在表格里标出 `classes_supported / classes_present`。

### 2.3 实例指标：oracle 与可部署要分开

| 字段 | 含义 | 可部署？ |
|---|---|---|
| 实例纯度（oracle） | 用 **GT 实例质心**做最近质心分类，算纯度 | ❌ 用了 GT，只衡量"嵌入空间有没有把实例分开" |
| 实例聚类 | 在嵌入的 kNN 图上按距离阈值取连通分量，再算碎片率/纯度 | ✅ 推理时就是这个路径 |

**只有后者能和几何跟踪器比** —— 它用的是和 `evaluate_against_gt.py`
完全相同的碎片率/纯度定义。前者是诊断量，不要放进结果表。

---

## 三、部署路径：ONNX 算子口径（`tools/rknn_op_audit.py`）

这里没有"精度"，只有**可行性**。每个 ONNX 算子按四档判定：

| 判定 | 含义 |
|---|---|
| `NPU` | 有对应支持，静态形状下可上 NPU |
| `SURGERY` | 不支持，但可等价改写成支持的算子 |
| `CPU_ONLY` | 必须留 CPU（动态索引 / 数据相关控制流） |
| `UNKNOWN` | 需真机实测（工具链版本相关） |

**两个坑**：

1. **两个导出器的节点数不可比。** TorchScript 导出器（`dynamo=False`，opset 12）
   是部署路径，图干净；dynamo 导出器（opset 17）分解得更碎。
   任何"节点数"数字都必须带上导出器标签。
2. **节点数 ≠ 耗时。** 一个 `GatherElements` 留在 CPU 的代价，
   和一个 256×512 的 `MatMul` 上 NPU 的代价，不是一个量级。
   节点计数只回答"图能不能切"，不回答"切完快不快"。

---

## 四、延迟口径（`tools/benchmark_latency.py`）

**只测推理，不测脚手架。** 具体做法：

- 单进程：模型（DINO / SAM2）只加载一次，不逐帧 spawn 子进程；
- 默认不落盘、不渲染预览（`--with-write` 才打开）；
- `--warmup N` 丢掉前 N 帧，避免 CUDA 首次编译与显存分配混进统计；
- 每段计时前后都 `torch.cuda.synchronize()` —— 否则测到的是 kernel 入队时间，不是执行时间。

报告每段的 mean / median / p95 / max。**用 median 而不是 mean 做结论**，
因为读帧偶尔会撞上磁盘抖动。

历史教训：早期口径（`run_instance_sequence.py`）是逐帧 spawn 子进程 + 写 PLY + 渲染预览，
得到 30.74 s/帧，其中约 90% 是脚手架开销。真实推理是 388.4 ms/帧。
**脚本自己的 docstring 里写了"本脚本总耗时不能作为实时性能指标"，但仍然被引用过 —— 所以现在单独做了基准。**
