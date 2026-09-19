"""开放词汇文本查询：用 CLIP 文本嵌入检索 3D 实例。

把自然语言查询编码为 CLIP 文本向量，与 extract_instance_embeddings.py
产出的实例图像嵌入做余弦相似度匹配，输出每个查询的排序结果。

三个关键设计：
1. 提示词集成（prompt ensembling）——同一文本用多个模板编码后平均，
   比单模板稳定；文本本身若已含冠词则跳过 "a photo of a {}" 模板。
2. 跨查询可比的 z-score——不同文本在 CLIP 空间中的相似度基线不同
   （例如 "chair" 整体相似度天然高于 "a place to sit"），
   因此除原始余弦值外，额外给出该查询在实例集合上的标准化分数。
3. 判定"命中"默认用概念词表指派（vocabulary 模式），而不是固定阈值：
   对每个实例，在候选概念词表上取 argmax，只有 argmax 命中该概念、
   且与次优概念的余弦差超过 margin 才接受。
   固定 z 阈值在这个数据上很脆弱——office0 只有 21 个实例、其中真椅子
   仅 2 个，分数分布被非椅子主导，导致 "chair" 的 z 最大值只有 1.43，
   任何全局阈值都会把它整类砍掉。
   自由文本查询（如 "a place to sit"）先解析到最相近的概念词，再复用
   该概念的实例指派，因此改写说法也能命中。

用法（在项目根目录）：
python -m tools.query_instances \
    --embedding-directory outputs/experiments/office0_fixed_detector_v3/embeddings \
    --queries "chair" "a place to sit" "trash can" \
    --top-k 5 \
    --output outputs/experiments/office0_fixed_detector_v3/embeddings/query_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.perception.open_vocab_encoder import OpenVocabularyEncoder

DEFAULT_TEMPLATES = ["{}", "a photo of a {}", "a photo of {}", "a close-up photo of a {}"]

# 与 src/mapping/geometric_instance_tracker.py 的 KNOWN_LABELS 保持一致，
# 这样文本查询与几何关联用的是同一套概念，结果可直接比较。
DEFAULT_VOCABULARY = ["chair", "desk", "door", "computer monitor", "trash can", "sofa"]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-directory", required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--accept-mode", choices=["vocabulary", "z_score", "top_k", "none"],
        default="vocabulary", help="判定命中的方式，默认概念词表指派",
    )
    parser.add_argument("--vocabulary", nargs="+", default=None,
                        help="候选概念词表，默认用检测器的 KNOWN_LABELS")
    # 默认 0.010 由 tools/evaluate_open_vocab_query.py 的 margin 扫描定标：
    # margin=0 时精度 0.46、0.005 时 0.52、>=0.010 时精度 1.00（召回 0.36）。
    # 0.010 是精度达到 1.0 的拐点，故取作默认。
    parser.add_argument("--vocabulary-margin", type=float, default=0.010,
                        help="实例 argmax 概念与次优概念的最小余弦差（默认 0.010）")
    parser.add_argument("--accept-z", type=float, default=1.5, help="z_score 模式的阈值")
    parser.add_argument("--accept-top-k", type=int, default=3, help="top_k 模式的接受数量")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def template_variants(text, templates):
    """生成提示词变体；文本已含冠词时不再叠加 "of a"。"""

    lowered = text.strip().lower()
    has_article = lowered.startswith(("a ", "an ", "the "))
    variants = []
    for template in templates:
        if has_article and "of a {}" in template:
            continue
        variants.append(template.format(text))
    return variants or [text]


def embed_texts(encoder, texts):
    """把一组文本（含模板集成）编码成单位向量矩阵，返回 (len(texts), D)。"""

    prompts, owner = [], []
    for index, text in enumerate(texts):
        for variant in template_variants(text, DEFAULT_TEMPLATES):
            prompts.append(variant)
            owner.append(index)

    features = encoder.encode_texts(prompts).numpy()
    owner = np.asarray(owner)

    vectors = []
    for index in range(len(texts)):
        stacked = features[owner == index]
        mean_vector = stacked.mean(axis=0)
        vectors.append(mean_vector / max(np.linalg.norm(mean_vector), 1e-8))
    return np.stack(vectors)


def load_embeddings(directory):
    directory = Path(directory)
    archive = np.load(directory / "instance_embeddings.npz")
    global_ids = archive["global_ids"].astype(int).tolist()
    embeddings = archive["embeddings"].astype(np.float32)
    metadata = json.load(open(directory / "instance_embeddings.json", encoding="utf-8"))
    info = {int(item["global_id"]): item for item in metadata["instances"]}
    return global_ids, embeddings, metadata, info


def main():
    args = parse_arguments()

    global_ids, embeddings, metadata, info = load_embeddings(args.embedding_directory)
    if metadata.get("clip_model") != args.clip_model:
        print("提示：嵌入由 %s 生成，本次查询使用 %s，两者必须一致才有意义"
              % (metadata.get("clip_model"), args.clip_model))

    vocabulary = args.vocabulary or DEFAULT_VOCABULARY
    encoder = OpenVocabularyEncoder(args.clip_model, device=args.device)

    # 概念词表与查询文本一次性编码
    vocabulary_features = embed_texts(encoder, vocabulary)
    query_features = embed_texts(encoder, args.queries)

    # (N, V) 实例-概念相似度矩阵
    instance_concept_raw = embeddings @ vocabulary_features.T

    # 概念偏置校正（关键）：CLIP 对不同词有不同的相似度基线，本例中
    # "trash can" 对几乎所有实例都偏高，直接 argmax 会把 G007/G009/G011/G013
    # 等一大半实例判成 trash can，而没有一个实例被判成 chair / desk。
    # 按列减去该概念在所有实例上的均值，即零样本分类里的 prior correction
    # （Calibrate-Before-Use），消除"哪个词整体相似度高"的影响，
    # 只保留"这个实例相对其他实例更符合哪个概念"。
    concept_bias = instance_concept_raw.mean(axis=0, keepdims=True)
    instance_concept = instance_concept_raw - concept_bias

    concept_order = np.argsort(-instance_concept, axis=1)
    best_concept = concept_order[:, 0]
    best_score = instance_concept[np.arange(len(global_ids)), best_concept]
    runner_up = (
        instance_concept[np.arange(len(global_ids)), concept_order[:, 1]]
        if instance_concept.shape[1] > 1 else np.full(len(global_ids), -np.inf)
    )
    concept_margin = best_score - runner_up

    # 查询 → 概念解析；同样按行做基线校正，否则所有查询都会解析到同一个词。
    query_concept_raw = query_features @ vocabulary_features.T
    query_concept = query_concept_raw - query_concept_raw.mean(axis=1, keepdims=True)
    query_concept_index = np.argmax(query_concept, axis=1)

    queries = []
    similarity_by_query = {}
    for q_index, query in enumerate(args.queries):
        vector = query_features[q_index]
        scores = embeddings @ vector
        similarity_by_query[query] = scores

        order = np.argsort(-scores)
        mean, std = float(scores.mean()), float(scores.std())
        z_scores = (scores - mean) / max(std, 1e-8)
        percentile = (np.argsort(np.argsort(scores)) + 1) / len(scores)

        ranking = []
        for rank, position in enumerate(order[: args.top_k], start=1):
            global_id = global_ids[position]
            entry = info.get(global_id, {})
            ranking.append({
                "rank": rank,
                "global_id": global_id,
                "score": round(float(scores[position]), 4),
                "z_score": round(float(z_scores[position]), 3),
                "percentile": round(float(percentile[position]), 3),
                "fused_label": entry.get("label", "unknown"),
                "crops_used": entry.get("crops_used", 0),
                "assigned_concept": vocabulary[int(best_concept[position])],
                "concept_margin": round(float(concept_margin[position]), 4),
            })

        resolved_concept = vocabulary[int(query_concept_index[q_index])]
        resolution_score = float(query_concept_raw[q_index, query_concept_index[q_index]])
        resolution_margin = float(
            query_concept[q_index, query_concept_index[q_index]]
            - np.partition(query_concept[q_index], -2)[-2]
        )

        if args.accept_mode == "vocabulary":
            accepted_mask = (best_concept == query_concept_index[q_index]) & (
                concept_margin >= args.vocabulary_margin
            )
            accepted = [global_ids[i] for i in np.where(accepted_mask)[0]]
        elif args.accept_mode == "z_score":
            accepted = [global_ids[i] for i in np.where(z_scores >= args.accept_z)[0]]
        elif args.accept_mode == "top_k":
            accepted = [global_ids[i] for i in order[: args.accept_top_k]]
        else:
            accepted = []
        accepted = sorted(accepted, key=lambda g: -scores[global_ids.index(g)])

        queries.append({
            "query": query,
            "prompts": template_variants(query, DEFAULT_TEMPLATES),
            "resolved_concept": resolved_concept,
            "resolution_score": round(resolution_score, 4),
            "resolution_margin": round(resolution_margin, 4),
            "statistics": {
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
                "mean": round(mean, 4),
                "std": round(std, 4),
            },
            "acceptance": {
                "mode": args.accept_mode,
                "vocabulary_margin": (args.vocabulary_margin
                                      if args.accept_mode == "vocabulary" else None),
                "z_threshold": args.accept_z if args.accept_mode == "z_score" else None,
            },
            "accepted": accepted,
            "ranking": ranking,
            # 全部实例的分数，供离线算 AP 用（top-k 之外的部分否则会丢失）。
            "scores": {str(global_ids[i]): round(float(scores[i]), 6)
                       for i in range(len(global_ids))},
        })

    # 每个实例最终归属哪个概念——与融合标签对照可看出语义模块是否与几何模块一致
    instance_assignment = []
    for position, global_id in enumerate(global_ids):
        candidates = [(q, similarity_by_query[q][position]) for q in similarity_by_query]
        top_query, top_score = max(candidates, key=lambda item: item[1])
        instance_assignment.append({
            "global_id": global_id,
            "fused_label": info.get(global_id, {}).get("label", "unknown"),
            "assigned_concept": vocabulary[int(best_concept[position])],
            "concept_margin": round(float(concept_margin[position]), 4),
            "best_query": top_query,
            "best_query_score": round(float(top_score), 4),
        })

    report = {
        "clip_model": args.clip_model,
        "embedding_directory": str(args.embedding_directory),
        "instance_count": len(global_ids),
        "templates": DEFAULT_TEMPLATES,
        "vocabulary": vocabulary,
        # 每个概念在所有实例上的平均相似度，即被校正掉的基线偏置；
        # 数值差异越大说明 CLIP 的词间偏置越严重。
        "concept_bias": {name: round(float(concept_bias[0, i]), 4)
                         for i, name in enumerate(vocabulary)},
        "queries": queries,
        "instance_assignment": instance_assignment,
        # 兼容旧字段名
        "instance_best_query": instance_assignment,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # 控制台表格
    print("概念基线偏置（已从相似度中扣除）：%s"
          % "  ".join("%s=%.3f" % (name, concept_bias[0, i])
                      for i, name in enumerate(vocabulary)))
    for entry in queries:
        print("\n查询：%s  ->  概念「%s」（解析相似度 %.3f）"
              % (entry["query"], entry["resolved_concept"], entry["resolution_score"]))
        print("  余弦 %.3f~%.3f（均值 %.3f）"
              % (entry["statistics"]["min"], entry["statistics"]["max"],
                 entry["statistics"]["mean"]))
        print("  %-4s %-6s %8s %8s  %-16s %-16s %s"
              % ("rank", "G-ID", "cosine", "z", "融合标签", "指派概念", "一致"))
        for row in entry["ranking"]:
            agrees = row["fused_label"].lower() == row["assigned_concept"].lower()
            print("  %-4d G%03d  %8.4f %8.3f  %-16s %-16s %s"
                  % (row["rank"], row["global_id"], row["score"], row["z_score"],
                     row["fused_label"], row["assigned_concept"], "OK" if agrees else ""))
        print("  接受（%s）：%s"
              % (entry["acceptance"]["mode"],
                 ", ".join("G%03d" % g for g in entry["accepted"]) or "无"))

    print("\n输出：%s" % args.output)


if __name__ == "__main__":
    main()
