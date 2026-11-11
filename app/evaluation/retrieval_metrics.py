"""检索质量指标（P1）。

纯函数实现 Recall@K / Hit@K / MRR，供离线 golden set 评测与单测使用。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Top-K 内是否命中任一相关项。"""
    if k <= 0 or not relevant:
        return 0.0
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Top-K 覆盖了多少比例的相关项（多标签查询用）。"""
    if k <= 0 or not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """第一个相关项排名的倒数。"""
    if not relevant:
        return 0.0
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_rankings(
    records: Sequence[Mapping[str, Any]],
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """按 golden set 汇总检索指标。

    Args:
        records: 每项至少包含 query / relevant / bucket。
        rankings: 与 records 对齐的检索结果（已按相关性排序的菜名列表）。
        k: Hit@K / Recall@K 的 K。
    """
    if len(records) != len(rankings):
        raise ValueError("records 与 rankings 长度必须一致")

    overall_hit1: list[float] = []
    overall_hitk: list[float] = []
    overall_recall: list[float] = []
    overall_rr: list[float] = []
    by_bucket: dict[str, dict[str, list[float]]] = {}
    misses: list[dict[str, Any]] = []

    for record, ranked in zip(records, rankings):
        query = str(record.get("query", ""))
        relevant = {str(item) for item in record.get("relevant", [])}
        bucket = str(record.get("bucket") or "default")

        hit1 = hit_at_k(ranked, relevant, 1)
        hitk = hit_at_k(ranked, relevant, k)
        recall = recall_at_k(ranked, relevant, k)
        rr = reciprocal_rank(ranked, relevant)

        overall_hit1.append(hit1)
        overall_hitk.append(hitk)
        overall_recall.append(recall)
        overall_rr.append(rr)

        bucket_stats = by_bucket.setdefault(
            bucket, {"hit1": [], "hitk": [], "recall": [], "rr": []}
        )
        bucket_stats["hit1"].append(hit1)
        bucket_stats["hitk"].append(hitk)
        bucket_stats["recall"].append(recall)
        bucket_stats["rr"].append(rr)

        if hitk == 0:
            misses.append(
                {
                    "query": query,
                    "bucket": bucket,
                    "relevant": sorted(relevant),
                    "ranked": list(ranked[:k]),
                }
            )

    return {
        "samples": len(records),
        "k": k,
        "hit_at_1": round(_mean(overall_hit1), 4),
        f"hit_at_{k}": round(_mean(overall_hitk), 4),
        f"recall_at_{k}": round(_mean(overall_recall), 4),
        "mrr": round(_mean(overall_rr), 4),
        "by_bucket": {
            bucket: {
                "samples": len(stats["hitk"]),
                "hit_at_1": round(_mean(stats["hit1"]), 4),
                f"hit_at_{k}": round(_mean(stats["hitk"]), 4),
                f"recall_at_{k}": round(_mean(stats["recall"]), 4),
                "mrr": round(_mean(stats["rr"]), 4),
            }
            for bucket, stats in sorted(by_bucket.items())
        },
        "misses": misses,
    }


__all__ = [
    "hit_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "evaluate_rankings",
]
