"""开放词汇文本查询：用 CLIP 文本嵌入检索 3D 实例。

把自然语言查询编码为 CLIP 文本向量，与 extract_instance_embeddings.py
产出的实例图像嵌入做余弦相似度匹配，输出每个查询的排序结果。

两个关键设计：
1. 提示词集成（prompt ensembling）——同一查询用多个模板编码后平均，
   比单模板稳定；查询本身若已含冠词则跳过 "a photo of a {}" 模板。
2. 跨查询可比的 z-score——不同文本在 CLIP 空间中的相似度基线不同
   （例如 "chair" 整体相似度天然高于 "a place to sit"），
   因此除原始余弦值外，额外给出该查询在实例集合上的标准化分数，
   判断"是否命中"应以 z-score 为准。

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


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-directory", required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--accept-mode", choices=["z_score", "top_k", "none"], default="z_score",
        help="判定命中的方式；z_score 用标准化分数，跨查询可比",
    )
    # 默认 1.5 由 tools/evaluate_open_vocab_query.py 的阈值扫描定标：
    # 在 office0 的 10 个查询上，z=1.0 精度 0.56、z=1.5 精度 0.79、z=2.0 精度 1.00。
    parser.add_argument("--accept-z", type=float, default=1.5,
                        help="z_score 模式的阈值（默认 1.5，兼顾精度与召回）")
    parser.add_argument("--accept-top-k", type=int, default=3, help="top_k 模式的接受数量")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def template_variants(query, templates):
    """生成提示词变体；查询已含冠词时不再叠加 "of a"。"""

    lowered = query.strip().lower()
    has_article = lowered.startswith(("a ", "an ", "the "))
    variants = []
    for template in templates:
        if has_article and "of a {}" in template:
            continue
        variants.append(template.format(query))
    return variants or [query]


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

    encoder = OpenVocabularyEncoder(args.clip_model, device=args.device)

    # 所有查询的提示词变体一次性编码，再按查询聚合，减少前向次数。
    prompts, owner = [], []
    for index, query in enumerate(args.queries):
        for variant in template_variants(query, DEFAULT_TEMPLATES):
            prompts.append(variant)
            owner.append(index)

    text_features = encoder.encode_texts(prompts).numpy()
    owner = np.asarray(owner)

    queries = []
    similarity_by_query = {}
    for index, query in enumerate(args.queries):
        stacked = text_features[owner == index]
        if len(stacked) == 0:
            continue
        mean_prompt = stacked.mean(axis=0)
        mean_prompt = mean_prompt / max(np.linalg.norm(mean_prompt), 1e-8)

        scores = embeddings @ mean_prompt
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
                "label_agrees": entry.get("label", "").lower() in query.lower()
                                 or query.lower() in entry.get("label", "").lower(),
            })

        if args.accept_mode == "z_score":
            accepted = [global_ids[i] for i in np.where(z_scores >= args.accept_z)[0]]
        elif args.accept_mode == "top_k":
            accepted = [global_ids[i] for i in order[: args.accept_top_k]]
        else:
            accepted = []
        accepted = sorted(accepted, key=lambda g: -scores[global_ids.index(g)])

        queries.append({
            "query": query,
            "prompts": template_variants(query, DEFAULT_TEMPLATES),
            "statistics": {
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
                "mean": round(mean, 4),
                "std": round(std, 4),
            },
            "acceptance": {"mode": args.accept_mode,
                           "threshold": args.accept_z if args.accept_mode == "z_score" else None},
            "accepted": accepted,
            "ranking": ranking,
            # 全部实例的分数，供离线算 AP 用（top-k 之外的部分否则会丢失）。
            "scores": {str(global_ids[i]): round(float(scores[i]), 6)
                       for i in range(len(global_ids))},
        })

    # 每个实例最匹配哪个查询——用于发现"查询没覆盖到"的实例。
    best_query = []
    for position, global_id in enumerate(global_ids):
        candidates = [(q, similarity_by_query[q][position]) for q in similarity_by_query]
        top_query, top_score = max(candidates, key=lambda item: item[1])
        best_query.append({
            "global_id": global_id,
            "fused_label": info.get(global_id, {}).get("label", "unknown"),
            "best_query": top_query,
            "score": round(float(top_score), 4),
        })

    report = {
        "clip_model": args.clip_model,
        "embedding_directory": str(args.embedding_directory),
        "instance_count": len(global_ids),
        "templates": DEFAULT_TEMPLATES,
        "queries": queries,
        "instance_best_query": best_query,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # 控制台表格
    for entry in queries:
        print("\n查询：%s   （余弦 %.3f~%.3f，均值 %.3f）"
              % (entry["query"], entry["statistics"]["min"],
                 entry["statistics"]["max"], entry["statistics"]["mean"]))
        print("  %-4s %-6s %8s %8s %6s  %-16s %s"
              % ("rank", "G-ID", "cosine", "z", "pct", "融合标签", "一致"))
        for row in entry["ranking"]:
            print("  %-4d G%03d  %8.4f %8.3f %6.2f  %-16s %s"
                  % (row["rank"], row["global_id"], row["score"], row["z_score"],
                     row["percentile"], row["fused_label"],
                     "OK" if row["label_agrees"] else ""))
        print("  接受（%s）：%s"
              % (entry["acceptance"]["mode"],
                 ", ".join("G%03d" % g for g in entry["accepted"]) or "无"))

    print("\n输出：%s" % args.output)


if __name__ == "__main__":
    main()
