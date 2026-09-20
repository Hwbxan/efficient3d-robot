# 结果汇总

每个数字都标了来源文件与复现命令。原始 JSON 都在 `docs/results/`。
指标定义见 [`METRICS.md`](METRICS.md)。

---

## 1. 在线感知（Replica office0，61 帧）

**数据**：`datasets/processed/Replica/office0`，帧 0–600 步长 10。
**来源**：`docs/results/gt_eval_office0.json`（实验目录 `office0_filtered_v5_sofa`）。

### 1.1 检测（逐帧 2D，GT 实例掩码）

| 指标 | IoU 0.25 | IoU 0.50 |
|---|---|---|
| **AP** | **0.7027** | **0.3142** |
| 精确率 | 0.6681 | 0.5240 |
| 召回率 | 0.8095 | 0.6349 |
| 真阳性 | 153 | 120 |
| 预测实例数 | 229 | 229 |
| GT 实例数 | 189 | 189 |

评测口径：`--min-gt-area 100`、`--max-area-fraction 0.4`、GT 类别过滤后 6 类。

### 1.2 关联（3D 跟踪）

| 指标 | 数值 | 说明 |
|---|---|---|
| 匹配上的 GT 物体 | 9 | |
| 预测轨道数 | 13 | **过分割的直接读数：13/9 = 1.44** |
| 单轨覆盖的 GT 物体 | 5 | |
| `single_track_ratio` | 0.5556 | 越高越好 |
| `pure_track_ratio` | 1.0000 | 没有一条轨道混进多个物体 |
| `fragmentation_mean` | 1.4444 | 1.0 是完美 |
| 语义标签一致性 | 117 / 120 = **0.975** | 只在已关联观测上算 |
| 未关联观测 | 5 | |

**结论**：轨道纯度是满分，问题全在**碎片化** —— 4 个 GT 物体被拆成了两条轨道
（sofa、door、table 各一，见 JSON 的 `fragmented_gt_objects`）。
根因在 2D 分割前端，与 OVI-MAP 论文 Table 5 的判断一致
（SAM2 → CropFormer 让 instance AP50 从 27.8 涨到 50.8，论文原话是 SAM2
"high recall but over-segmentation"）。

**复现**：
```bash
python -m tools.evaluate_against_gt \
    --tracking-json outputs/experiments/office0_filtered_v5_sofa/association/tracking.json \
    --segmentation-root outputs/experiments/office0_filtered_v5_sofa/segmentation \
    --gt-root outputs/gt/office0 \
    --output outputs/experiments/office0_filtered_v5_sofa/gt_eval_all.json \
    --max-area-fraction 0.4
```

---

## 1.5 多场景几何评测（8 个 Nice-SLAM 标准序列）

**数据**：Replica 的 8 个社区标准 RGB-D 序列（office_0–4、room_0–2），
每场景 200 帧（0–1990，步长 10），从官方语义网格反投影出 GT 实例掩码。
**来源**：`outputs/multiscene_eval_v2/summary.json`（box-threshold 0.30）。

| 场景 | 帧 | GT 实例 | AP@.25 | AP@.50 | 单轨率 | 纯轨率 | 碎片 | 标签一致 |
|---|---|---|---|---|---|---|---|---|
| office_0 | 200 | 1167 | 0.5825 | 0.3336 | 0.667 | 0.786 | 1.417 | 0.998 |
| office_1 | 200 | 575 | 0.5116 | 0.4853 | 0.667 | 1.000 | 1.333 | 0.990 |
| office_2 | 200 | 1302 | 0.6237 | 0.4896 | 0.800 | 0.944 | 1.267 | 0.970 |
| office_3 | 200 | 1774 | 0.7024 | 0.6554 | 0.533 | 0.809 | 1.800 | 0.997 |
| office_4 | 200 | 1069 | 0.7057 | 0.5332 | 0.533 | 0.789 | 1.533 | 0.988 |
| room_0 | 200 | 1207 | 0.6456 | 0.6153 | 0.846 | 0.846 | 1.154 | 0.999 |
| room_1 | 200 | 101 | 0.2069 | 0.1578 | 1.000 | 1.000 | 1.000 | 0.935 |
| room_2 | 200 | 1065 | 0.8887 | 0.8785 | 0.600 | 0.923 | 1.400 | 0.998 |
| **宏平均** | | | **0.6084** | **0.5186** | **0.706** | **0.887** | **1.363** | **0.984** |

room_1 的 GT 实例只有 101 个且几乎都是墙/地/天花（可见家具极少），AP 偏低属正常。
**碎片率从单场景老口径的 1.44 降到 1.36、单轨率从 0.556 升到 0.706**，关键改进是
把关联判据从 IoU 换成**体素覆盖率**（同一物体覆盖率 ~0.7+、不同物体 ~0，间隔干净），
修掉了「视角变化导致质心漂移」把同一物体切成两条轨道的问题。

**复现**：
```bash
python -m tools.run_multiscene_eval --all --real-only --skip-render \
    --output-root outputs/multiscene_eval_v2
```

### 1.5.1 指标口径：为什么「数字还行」和「视频观感一般」不矛盾

这组数字被反复问到「是不是只在单帧上算的」，把口径完整写在这里。

**(1) 不是单帧，规模不小。** 上表是 8 场景 × 每场景 200 帧（0–1990，步长 10）的聚合，
共 **8260 个 GT 实例 / 7406 个预测**，不是单帧或少数几帧的结果。

**(2) 但确实是「逐帧独立」评估。** 每一帧的预测单独与该帧 GT 掩码匹配、
再跨帧聚合出 AP/P/R。它衡量的是**单帧 2D 检测+分割质量**，
**完全不衡量跨帧一致性**：同一物体在相邻帧间掩码忽大忽小、颜色跳变、
两个物体被合并成一个 —— 这些在指标里都看不见，但在视频里一秒 24 帧地暴露出来。

**(3) headline 报的是宽松门槛。** 宏平均 AP@0.25 = 0.609 看着体面，
但 AP@0.50 只有 0.519 —— 相当比例的实例掩码 IoU 卡在 0.25–0.5 之间，
被算作「命中」，肉眼却能明显看出边缘不准/偏大/残缺。视频里看到的正是这部分。

**(4) 做视频挑的是最差的场景之一。** office_0 的 AP@0.50 = 0.334，
8 个场景里排**倒数第二**（仅高于 room_1 的 0.158，而 room_1 只有 101 个 GT 实例、
几乎全是墙/地/天花）。room_2 高达 0.878。**拿 office_0 当门面是最坏选择。**

**(5) 连续帧比稀疏帧更差。** 同场景同 25 帧（0–240）对照：

| 配置 | 预测 | 命中 | P@0.50 | R@0.50 |
|---|---|---|---|---|
| 步长 10（稀疏，v2） | 130 | 81 | **0.623** | 0.566 |
| 步长 1（连续，dense） | 156 | 80 | **0.513** | 0.559 |

连续跑时预测数 +26、命中数持平，说明密集帧下跨帧确认更快放行，**误检变多、
precision 掉 11 个点**。视频用的是步长 1 的连续帧，所以比 headline 数字还要差一档。
（dense 版虽然渲染了 241 帧分割，GT 打分仍只在其间 25 帧上做，与上表口径一致。）

**(6) 同管线换场景的对照（已出片）。** room_2 用完全相同的参数跑步长 1 的连续帧，
25 个评估帧上 **AP@0.25 = AP@0.50 = 0.891**（P 0.806 / R 0.915）——
两个门槛数值相等，意味着**所有命中实例的 IoU 都 ≥ 0.5**，没有「勉强命中」的那批。
对照 office_0 同配置：AP@0.25 = 0.680 但 **AP@0.50 = 0.240**。
两段视频在线上站点并列，同一条管线、零参数差异，观感差距一眼可见。

**结论**：指标与观感并不冲突，是同一个事实的两面 —— 我们的**召回不错、
单帧定位大致对，但掩码质量（AP@0.50）是真实瓶颈**，而视频把这部分放大了。
诚实定位：离 SOTA 仍有明显差距，根因在 2D 分割前端（与 OVI-MAP Table 5 结论一致），
换检测器已验证无效（见 §5.5），要治本得换分割器（SAM2 → CropFormer 量级）。

---

## 1.6 开放词汇文本检索（CLIP）

每个融合 3D 实例的逐帧掩码裁剪送 CLIP（`openai/clip-vit-base-patch32`）图像编码器，
按掩码面积加权平均得实例嵌入；文本查询编码后余弦排序。用 GT 语义类别（与检测器
词表解耦）反推每个实例的真实类别，算检索 AP。评测 8 场景、6 个常见查询词。

**按「物体确实存在的场景」宏平均**（避免把没有该物体的场景记 0 分）：

| 查询 | 存在场景 AP | 含无物体场景 AP |
|---|---|---|
| chair | **0.840** | 0.735 |
| trash can | **0.818** | 0.716 |
| sofa | 0.769 | 0.385 |
| desk | 0.734 | 0.734 |
| door | 0.625 | 0.625 |
| computer monitor | **0.229** | 0.143 |
| **平均** | **0.669** | 0.556 |

computer monitor 弱的原因**不是检测器漏检**（此处更正 README 此前的误判）：

| 场景 | monitor AP | hits / GT | recall | 真显示器排名 |
|---|---|---|---|---|
| office_0 | **1.000** | 1 / 1 | 1.0 | 第 1 |
| office_1 | 0.143 | 1 / 1 | 1.0 | 第 7（p@5 = 0） |
| 其余 6 个场景 | — | 0 / **0** | — | 场景内无显示器 |

8 个场景**总共只有 2 个 GT 显示器**，且两个都成功进入地图（recall 1.0）。差距来自
CLIP 图文检索的**排序**：office_0 排第 1，office_1 掉到第 7。因此这是检索排序问题，
换检测器或加多尺度均无济于事（见下节实验）。其余 5 类达到 0.6–0.84，
作为「可文本查询的 3D 实例地图」已可用。

**复现**：
```bash
python -m tools.run_multiscene_open_vocab --scenes office_0 office_1 office_2 \
    office_3 office_4 room_0 room_1 room_2 --eval-root outputs/multiscene_eval_v2
```

---

## 2. 推理延迟（RTX 3090，单进程）

**来源**：`docs/results/benchmark_latency.json`（不落盘）、
`docs/results/benchmark_latency_io.json`（落盘 + 预览）。

### 2.1 纯推理（61 帧，warmup 3，n = 58）

| 阶段 | 均值 | 中位数 | p95 | max |
|---|---|---|---|---|
| 读帧 | 30.5 ms | 28.7 ms | 40.9 ms | 48.8 ms |
| Grounding DINO（2D 检测） | 207.2 ms | 203.5 ms | 247.9 ms | 259.7 ms |
| SAM 2（实例分割） | 92.0 ms | 88.1 ms | 130.3 ms | 152.1 ms |
| 深度反投影 | 58.6 ms | 52.2 ms | 104.7 ms | 147.3 ms |
| **合计** | **388.4 ms** | **392.6 ms** | 471.3 ms | 520.3 ms |

**2.57 FPS（旧 benchmark 口径）。** 纯 2D 推理（检测 + 分割）= 299.2 ms，占 **77.0%**。
GPU 显存占用约 1.8 GB。

> **当前流水线实测更快**：`run_sequence_efficient.py` 实测单帧 **162.5 ms → 6.15 FPS**
> （Grounding DINO 135.6 ms + SAM2 17.2 ms + 3D 提升 9.8 ms），见 README §1b。
> 上面的 388 ms 是更早的 `benchmark_latency.py` 口径（未开 fp16、SAM 模型更大），
> 保留作脚手架消除前后的对照。

### 2.2 加上落盘与预览（10 帧，warmup 2，n = 8）

| 阶段 | 均值 | 中位数 |
|---|---|---|
| 推理四段合计 | 406.7 ms | 398.4 ms |
| **写盘 + 渲染预览** | **1671.7 ms** | 1586.9 ms |
| 合计 | 2078.4 ms | 2001.6 ms |

**结论**：加上可视化后慢了 5.3×，其中 **80.4% 是预览渲染**。
这说明剩余开销**全在可视化，不在推理** —— 也解释了旧的 30.74 s/帧 是怎么来的。

### 2.3 与旧口径的对比

| 口径 | 每帧耗时 | 说明 |
|---|---|---|
| `run_instance_sequence.py`（旧） | 30.74 s | 逐帧 spawn 子进程、重复加载权重、写 PLY、渲染预览 |
| `benchmark_latency.py`（新） | **388.4 ms** | 单进程、模型只加载一次、不落盘 |

**79× 的差距全部来自脚手架**，不是模型变快了。
（旧脚本的 docstring 里就写了"本脚本总耗时不能作为实时性能指标"，但仍被引用过。）

**复现**：
```bash
python -m tools.benchmark_latency --frames 0 10 20 30 40 50 60 --warmup 3 \
    --output outputs/benchmark_latency.json
python -m tools.benchmark_latency --frames 0 10 20 30 --with-write --warmup 2 \
    --output outputs/benchmark_latency_io.json
```

---

## 3. Stage 5 点式骨干（跨场景）

**划分按场景，不按块** —— 验证/测试用的都是训练时**没见过的房间**。

| 划分 | 场景 | 用途 |
|---|---|---|
| train | 12 个（apartment_0/1/2、frl_apartment_0–5、hotel_0、office_0/1） | 训练 |
| val | office_2 / office_3 / office_4 | 选模型 |
| test | room_0 / room_1 / room_2 | **从未参与任何决策** |

**来源**：`docs/results/eval_val.json`、`docs/results/eval_test.json`。

### 3.1 总体

| 划分 | 点数 | 准确率 | mIoU | 宏平均 mIoU | 支撑度≥200 的 mIoU | 频次加权 IoU |
|---|---|---|---|---|---|---|
| val | 591,964 | 0.5398 | 0.1517 | 0.1983 | 0.1863 (22/30 类) | 0.4769 |
| **test** | 593,653 | **0.5872** | **0.1632** | **0.2249** | **0.1927 (32/38 类)** | 0.4734 |

**测试集比验证集更好**，说明没有在验证集上过拟合。测试集（房间类场景）
出现的类别还更多（38 vs 30 类），说明这个优势不是靠类别少得来的。

### 3.2 逐场景

| 场景 | 标注点数 | 准确率 | mIoU | 支撑度≥200 的 mIoU |
|---|---|---|---|---|
| office_2 | 196,339 | 0.5958 | 0.2375 | 0.3359 (14 类) |
| office_3 | 197,957 | 0.5282 | 0.1471 | 0.2116 (17 类) |
| office_4 | 197,668 | 0.4957 | 0.2102 | 0.2602 (13 类) |
| room_0 | 198,287 | 0.6531 | 0.2450 | 0.2859 (24 类) |
| room_1 | 198,469 | 0.5583 | 0.2169 | 0.2409 (21 类) |
| room_2 | 196,897 | 0.5499 | 0.2128 | 0.2473 (18 类) |

### 3.3 逐类（IoU 前 8）

| 划分 | 表现最好的类别 |
|---|---|
| val | ceiling 0.882、lamp 0.675、floor 0.653、wall 0.520、camera 0.278、bin 0.267、table 0.243、vent 0.239 |
| test | ceiling 0.902、rug 0.555、lamp 0.550、floor 0.538、wall 0.489、table 0.471、chair 0.384、indoor-plant 0.304 |

规律很清楚：**结构面（ceiling / floor / wall）+ 大件家具（table / chair / rug）好，
长尾小物体差**。这就是宏平均 mIoU 被拖低的原因，也是必须同时报
`miou_supported` 与 `frequency_weighted_iou` 的原因。

**复现**：
```bash
python -m tools.build_mesh_point_dataset --output-directory outputs/point_dataset
python -m tools.train_point_encoder --dataset-root outputs/point_dataset \
    --output-directory outputs/stage5/run_s5b_metricfix \
    --epochs 60 --num-points 8192 --lr 2e-3 --class-weight-mode inverse_sqrt
python -m tools.evaluate_point_encoder --checkpoint outputs/stage5/run_s5b_metricfix/best.pt \
    --dataset-root outputs/point_dataset --split test \
    --output outputs/stage5/run_s5b_metricfix/eval_test.json
```

---

## 4. RKNN 算子兼容性审计

**来源**：`docs/results/rknn_op_audit.json`。**不需要任何硬件。**

### 4.1 三档图切分

| 档位 | 导出器 | 总节点 | 可上 NPU | 需手术 | 必须留 CPU |
|---|---|---|---|---|---|
| 完整图（含 Morton + kNN） | TorchScript opset 12 | **导出失败** | — | — | — |
| 完整图（仅作量级参考） | dynamo opset 17 | 3425 | 2662 (77.7%) | 26 (0.8%) | **678 (19.8%)** |
| Morton 移出 | TorchScript opset 12 | 719 | 665 (92.5%) | 30 (4.2%) | **24 (3.3%)** |
| Morton + kNN 移出 | TorchScript opset 12 | 639 | 594 (93.0%) | 30 (4.7%) | **15 (2.3%)** |

同一导出器内可比的两组：dynamo 口径 678 → 27 → 18；
部署路径口径 导不出来 → 24 → 15。结论一致：**负担几乎全来自 Morton 编码**。

### 4.2 完整图为什么导不出来

用部署路径的导出器，**每一个 opset 都失败**：

| opset | 结果 |
|---|---|
| 12 / 13 / 15 / 17 | `ONNX export does NOT support exporting bitwise AND for non-boolean input values` |
| 18 / 19 / 20 | `ONNX export does NOT support exporting bitwise OR for non-boolean input values` |

即 Morton 的位运算**连 ONNX 这一关都过不去**，不需要等到上 NPU 才被拒绝。
换成 dynamo 导出器可以导出来，位运算族合计 630 个节点（BitShift 270 +
BitwiseOr 180 + BitwiseAnd 90 + BitwiseNot 90），另有 Cast 765 个。

### 4.3 部署图剩下的算子

剥离 Morton + kNN 后只剩 **`GatherElements` ×15**（`gather_features`，数据相关索引，留 CPU 合理）。
"需手术"恰好 **30 个节点 = `Erf` 13 + `Sqrt` 13 + `ReduceL2` 4**。

### 4.4 参数量分布

| 项 | 值 |
|---|---|
| Gemm 参数 | **970,215** |
| 总参数 | 975,975 |
| 占比 | **99.4%** |

压缩的主战场非常集中 —— 只有 16 个 `nn.Linear`。

**复现**：`python -m tools.rknn_op_audit --json outputs/experiments/rknn_op_audit.json`

---

## 5. RKNN 可部署形态（改写后）

**来源**：`docs/results/rknn_export.json`。

### 5.1 替换的模块

| 原实现 | 替换为 | 消除的算子 | 实测偏差 |
|---|---|---|---|
| `nn.GELU()` | `RKNNFriendlyGELU`（tanh 近似） | `Erf` | **4.7e-4**（唯一非严格等价） |
| `nn.LayerNorm` | `RKNNFriendlyLayerNorm`（`Pow(·,-0.5)`） | `Sqrt` | **4.8e-7** |
| `L2Normalise`（`F.normalize`） | `RKNNFriendlyL2Normalise` | `ReduceL2` / `Sqrt` | **6.0e-8** |
| `InverseDistanceWeight`（`.norm()`） | `RKNNFriendlyInverseDistanceWeight` | `ReduceL2` / `Sqrt` / `Reciprocal` | **0** |

替换数量：GELU ×13、LayerNorm ×13、L2Normalise ×1、反距离加权 ×3。

### 5.2 整模型数值等价

| 输出 | max_abs | max_rel | cos_min |
|---|---|---|---|
| semantic_logits | 8.061e-04 | 5.507e-04 | 1.000000 |
| instance_embedding | 8.137e-04 | 6.545e-04 | 0.999999 |
| text_embedding | 1.058e-04 | 5.245e-04 | 1.000000 |
| **语义 argmax 一致率** | | | **99.951%** |

### 5.3 导出结果

| 图 | 导出器 | 节点数 | 算子种类 | 禁用算子 |
|---|---|---|---|---|
| 完整图 | TorchScript opset 12 | **导出失败** | — | — |
| 完整图（仅测量） | dynamo opset 17 | 3607 | 38 | `Sqrt` ×9（来自 kNN 的 `cdist`） |
| **部署图** | **TorchScript opset 12** | **705** | **22** | **0** ✅ |

部署图的 22 类算子：`Constant`、`ConstantOfShape`、`Mul`、`Add`、`Sub`、`Div`、
`Pow`、`MatMul`、`ReduceMean`、`ReduceSum`、`ReduceMax`、`Expand`、`Reshape`、
`Unsqueeze`、`Concat`、`Slice`、`Clip`、`Identity`、`Equal`、`Where`、`Tanh`、`GatherElements`。

### 5.4 三条工程保证（由冒烟测试断言）

1. **原模型不被改动** —— 42 个 `state_dict` 项逐个比对未变，模块类型逐个比对未变。
2. **checkpoint 双向兼容** —— 替换模块都不带参数，`state_dict` 键完全一致，
   `strict=True` 双向加载成功。
3. **部署图零禁用算子** —— `Erf` / `Sqrt` / `ReduceL2` 全为 0，且 22 类算子全部在支持表内。

**复现**：
```bash
python -m tools.rknn_export --onnx-out deploy/point_encoder_rknn.onnx \
    --json outputs/experiments/rknn_export.json
python -m tools._smoke_test_rknn_export
```

---

## 5.5 多尺度检测消融（Path C，结论：不采用）

**动机**：改善 computer monitor 这类小物体的检测召回。

**关键实现细节**：不能靠「先放大输入图再送入检测器」——Grounding DINO 的 processor
默认 `shortest_edge=800`，会把任意尺寸输入**重新归一化**，放大输入图会被抵消掉。
真正有效的做法是覆盖 processor 的 `size`（实测 800×1066 → 1200×1600 生效）。
已把该能力以 `multi_scale_sizes` 参数加入 `GroundingDinoDetector.predict`，
默认为 `None`，行为与单尺度完全一致（不改动既有指标）。

**实验设置**（office_1，均匀采样 60 帧，box-threshold 0.30，max-area 0.40）：

| 指标 | baseline（短边 800） | +1200 短边 pass | 变化 |
|---|---|---|---|
| computer monitor 检出数 | 18 | 17 | **−1** |
| 有 monitor 检出的帧数 | 10 / 60 | 10 / 60 | 0 |
| 每帧平均 monitor 数 | 0.300 | 0.283 | −0.017 |
| chair / desk / door / sofa 检出数 | 6 / 11 / 11 / 3 | 6 / 11 / 11 / 3 | 0 |
| trash can 检出数 | 16 | 14 | −2 |
| **单帧耗时** | **164 ms** | **453 ms** | **×2.75** |

**结论**：提高检测分辨率**没有带来任何召回增益**，反而因跨尺度 NMS 略微减少检出，
代价是 2.75 倍延迟。叠加 §1.6 里「monitor 召回本就是 100%」的事实，以及此前
Grounding DINO base 零收益的测试，判定：**不更换检测器、不启用多尺度**。
monitor 的短板是 CLIP 检索排序问题，应在检索侧（而非检测侧）解决。

**复现**：
```bash
python -m tools.ablate_multiscale_detection --scene office_1 --frames 60 \
    --extra "1200:2000" --out outputs/multiscale_ablation_office1.json
```

---

## 6. 还没做的

诚实列出，避免把静态分析当成实测：

- **RK3588 真机未跑。** §4、§5 全部是 ONNX 静态分析 + 数值等价验证。
  真实能不能转、跑多快、INT8 掉多少点，都还不知道。
- **功耗未测。** 需要 USB-C 功率计。
- **在线流水线只在 8 个 Nice-SLAM 场景评过。** 其余 10 个 Replica 场景
  （apartment / frl / hotel）只有 mesh/semantic，需自渲染 RGB-D 才能纳入 headline。
- **computer monitor 检索弱 —— 排序问题，非漏检。** 两场景 recall 均为 1.0，
  但 office_1 真显示器排第 7。**多尺度检测实验（Path C）已证否**：检测分辨率
  800→1200 短边后显示器检出 18→17、耗时 ×2.75，故不采用（详见下节）。
- **端到端延迟未在边缘设备上测。** 桌面 GPU 已到 162.5 ms/帧（6.15 FPS）实时档，
  但这是 RTX 3090 的数字，不等于 RK3588 上的表现。
