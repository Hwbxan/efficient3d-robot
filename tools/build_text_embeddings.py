"""为数据集类别生成 CLIP 文本嵌入，供 Stage 5c 的开放词汇头对齐。

**顺序即契约**：训练时 `text_alignment_loss` 用 `class_id` 去索引文本嵌入，
所以嵌入的行顺序**必须**与数据集 `manifest.json` 的 `class_names` 完全一致。
因此本工具直接从数据集清单读类别名（而不是让调用方另传一份），
并把顺序写进 npz，训练脚本再校验一次——顺序错位是那种"能跑完、指标烂、
但看不出哪里错"的失败。

用法（在项目根目录）：

    python -m tools.build_text_embeddings \
        --dataset-root outputs/point_dataset \
        --output outputs/stage5/text_embeddings.npz
"""

import argparse
from pathlib import Path

import numpy as np

# 与 Stage 4 的 query_instances.py 保持同一套模板，保证两处语义空间可比。
DEFAULT_TEMPLATES = ("{}", "a photo of a {}", "a photo of {}", "a close-up photo of a {}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", default="outputs/point_dataset",
                        help="用于读取 class_names 顺序的数据集目录")
    parser.add_argument("--output", default="outputs/stage5/text_embeddings.npz")
    parser.add_argument("--encoder-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--templates", nargs="+", default=list(DEFAULT_TEMPLATES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report-similarity", action="store_true",
                        help="打印最相似的类别对，用于发现近似重复的类别名")
    return parser.parse_args()


def read_class_names(dataset_root):
    """从数据集清单里取类别名，顺序即嵌入的行顺序。"""

    import json

    manifest_path = Path(dataset_root) / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到 {manifest_path}；先跑 tools.build_point_dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest.get("class_names")
    if not names:
        raise SystemExit(f"{manifest_path} 里没有 class_names")
    return list(names)


def expand_prompts(class_names, templates):
    """把每个类别名套上所有模板，返回 (提示词列表, 每类的提示词数量)。"""

    prompts = []
    for name in class_names:
        for template in templates:
            prompts.append(template.format(name))
    return prompts, len(templates)


def embed_class_names(class_names, encode_texts, templates=DEFAULT_TEMPLATES,
                      batch_size=32):
    """对每个类别做**提示词集成**：多个模板的嵌入取平均后再 L2 归一化。

    单模板对措辞很敏感（Stage 4 实测：直接类别词 P@1 100%、多词改写 0%），
    集成几个模板能显著稳住表现。
    """

    if not class_names:
        raise ValueError("class_names 为空")
    prompts, per_class = expand_prompts(class_names, templates)
    raw = np.asarray(encode_texts(prompts, batch_size), dtype=np.float32)
    if raw.shape[0] != len(prompts):
        raise ValueError(f"编码器返回 {raw.shape[0]} 条，期望 {len(prompts)} 条")

    reshaped = raw.reshape(len(class_names), per_class, raw.shape[-1])
    # 先按模板归一化再平均，避免"模长大的模板"主导结果
    norms = np.linalg.norm(reshaped, axis=-1, keepdims=True)
    averaged = (reshaped / np.maximum(norms, 1e-8)).mean(axis=1)
    final_norms = np.linalg.norm(averaged, axis=-1, keepdims=True)
    return (averaged / np.maximum(final_norms, 1e-8)).astype(np.float32)


def closest_class_pairs(embeddings, class_names, top=8):
    """找出余弦相似度最高的类别对，用于发现近似重复的类别名。"""

    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    pairs = []
    for index in range(len(class_names)):
        best = int(np.argmax(similarity[index]))
        pairs.append((float(similarity[index, best]), class_names[index], class_names[best]))
    pairs.sort(reverse=True)
    return pairs[:top]


def main():
    args = parse_arguments()
    class_names = read_class_names(args.dataset_root)
    print(f"从 {args.dataset_root} 读到 {len(class_names)} 个类别")

    from src.perception.open_vocab_encoder import OpenVocabularyEncoder

    encoder = OpenVocabularyEncoder(args.encoder_model, device=args.device)
    print(f"编码器 {args.encoder_model}  维度 {encoder.embedding_dim}  设备 {encoder.device}")

    def encode_texts(prompts, batch_size):
        return encoder.encode_texts(prompts, batch_size=batch_size)

    embeddings = embed_class_names(
        class_names, encode_texts, templates=tuple(args.templates),
        batch_size=args.batch_size,
    )
    print(f"嵌入矩阵 {embeddings.shape}，每类 {len(args.templates)} 个模板集成")

    norms = np.linalg.norm(embeddings, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise SystemExit(f"嵌入未归一化（范数范围 {norms.min():.4f}–{norms.max():.4f}）")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        class_names=np.array(class_names, dtype=object),
        embeddings=embeddings,
        templates=np.array(args.templates, dtype=object),
        encoder_model=args.encoder_model,
    )

    if args.report_similarity:
        print("\n最相似的类别对（余弦）：")
        for similarity, left, right in closest_class_pairs(embeddings, class_names):
            print("  %.4f  %-24s ~ %s" % (similarity, left, right))

    print(f"\n已写出 {output_path}")


if __name__ == "__main__":
    main()
