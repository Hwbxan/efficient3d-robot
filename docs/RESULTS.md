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

**2.57 FPS。** 纯 2D 推理（检测 + 分割）= 299.2 ms，占 **77.0%**。
GPU 显存占用约 1.8 GB。

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

| 场景 | 点数 | 准确率 | mIoU | 支撑度≥200 的 mIoU |
|---|---|---|---|---|
| office_2 | 196,339 | 0.5958 | 0.2375 | 0.3359 (14 类) |
| office_3 | — | 0.5282 | 0.1471 | 0.2116 (17 类) |
| office_4 | — | 0.4957 | 0.2102 | 0.2602 (13 类) |
| room_0 | 198,287 | 0.6531 | 0.2450 | 0.2859 (24 类) |
| room_1 | — | 0.5583 | 0.2169 | 0.2409 (21 类) |
| room_2 | — | 0.5499 | 0.2128 | 0.2473 (18 类) |

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

## 6. 还没做的

诚实列出，避免把静态分析当成实测：

- **RK3588 真机未跑。** §4、§5 全部是 ONNX 静态分析 + 数值等价验证。
  真实能不能转、跑多快、INT8 掉多少点，都还不知道。
- **功耗未测。** 需要 USB-C 功率计。
- **在线流水线只有单场景。** 服务器上只有 office0 有 RGB-D + 位姿序列。
- **开放词汇分支未评测。** 文本查询路径还没跑通定量评测。
- **端到端延迟未在边缘设备上测。** 388.4 ms/帧是 RTX 3090 的数字。
