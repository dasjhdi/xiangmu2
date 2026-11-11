"""检索 golden set 离线评测（P1）。

用法：

    # 当前配置的基线（dense 0.8 / BM25 0.2，top10，不重排）
    python -m scripts.eval_retrieval

    # 对比不同 hybrid 权重
    python -m scripts.eval_retrieval --weights 0.7,0.3 --label w07
    python -m scripts.eval_retrieval --weights 0.6,0.4 --label w06
    python -m scripts.eval_retrieval --ranker rrf --label rrf

    # 只跑前 30 条，快速验证
    python -m scripts.eval_retrieval --limit 30

指标：Hit@1 / Hit@K / Recall@K / MRR，按 exact / paraphrase / category 分桶。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

from langchain_core.documents import Document

from app.config import DefaultRAGConfig, settings
from app.evaluation.retrieval_metrics import evaluate_rankings
from app.rag.embeddings.embedding_factory import get_embedding_model
from app.rag.pipeline.query_policy import is_explicit_recipe_query
from app.rag.pipeline.retrieval import RetrievalOptimizationModule
from app.rag.vector_stores.vector_store_factory import get_vector_store


def load_golden(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(item, dict) or not item.get("query"):
                raise SystemExit(f"第 {line_number} 行缺少 query")
            relevant = item.get("relevant") or []
            if isinstance(relevant, str):
                relevant = [relevant]
            item["relevant"] = [str(value) for value in relevant if value]
            item.setdefault("bucket", "default")
            records.append(item)
    return records


def dish_names(documents: Sequence[Document]) -> list[str]:
    """按检索顺序去重出菜名，跳过索引页。"""
    names: list[str] = []
    seen: set[str] = set()
    for doc in documents:
        name = str((doc.metadata or {}).get("dish_name") or "").strip()
        if not name or name == "菜谱索引" or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def corpus_names() -> list[str]:
    """全量菜名，用于模拟线上「明确查询」判定。"""
    return sorted(
        item.stem for item in Path("data/HowToCook/dishes").rglob("*.md")
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * quantile))
    return round(ordered[max(0, min(index, len(ordered) - 1))], 1)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = DefaultRAGConfig
    retrieval = config.retrieval
    records = load_golden(Path(args.golden))
    if args.offset > 0:
        records = records[args.offset :]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit("golden set 为空")

    ranker = args.ranker or retrieval.ranker_type
    weights = (
        [float(value) for value in args.weights.split(",")]
        if args.weights
        else list(retrieval.ranker_weights)
    )
    if ranker == "weighted" and len(weights) != 2:
        raise SystemExit("--weights 需要两个值，例如 0.7,0.3")

    # RRF 的分数量纲和 weighted 不同，用 0.2 的阈值会全部过滤掉
    threshold = (
        args.threshold
        if args.threshold is not None
        else (retrieval.score_threshold if ranker == "weighted" else 0.0)
    )

    embeddings = get_embedding_model(config)
    vector_store = get_vector_store(
        milvus_config=settings.database.milvus,
        collection_name=config.vector_store.collection_names["recipes"],
        embeddings=embeddings,
        chunks=[],
        force_rebuild=False,
    )
    module = RetrievalOptimizationModule(
        vectorstore=vector_store,
        score_threshold=threshold,
        default_ranker_type=ranker,
        default_ranker_weights=weights,
    )

    reranker = None
    rerank_budget = 0
    if args.rerank:
        from app.rag.rerankers.lexical_reranker import (
            FaultTolerantReranker,
            LexicalReranker,
        )
        from app.rag.rerankers.siliconflow_reranker import SiliconFlowReranker

        # 与线上一致：远程失败/返回空时退回本地词法重排
        reranker_config = (
            config.reranker.model_copy(update={"model_name": args.rerank_model})
            if args.rerank_model
            else config.reranker
        )
        reranker = FaultTolerantReranker(
            SiliconFlowReranker(reranker_config), LexicalReranker()
        )
        rerank_budget = args.rerank_limit if args.rerank_limit > 0 else len(records)

    rankings: list[list[str]] = []
    latencies: list[float] = []
    reranked = 0
    known_names = corpus_names() if args.skip_rerank_explicit else []

    for index, record in enumerate(records):
        started = time.perf_counter()
        docs, _scores = await module.hybrid_search(
            record["query"],
            top_k=args.top_k,
            ranker_type=ranker,
            ranker_weights=weights,
            score_threshold=threshold,
        )
        if (
            reranker is not None
            and index < rerank_budget
            and not (
                args.skip_rerank_explicit
                and is_explicit_recipe_query(record["query"], known_names)
            )
        ):
            try:
                # 与线上一致：先用 max_candidates 截断，再按 top_n 保留
                docs = docs[: config.reranker.max_candidates]
                docs = await reranker.rerank(record["query"], docs)
                docs = docs[: config.reranker.top_n]
                reranked += 1
            except Exception as exc:  # pragma: no cover - 网络问题不应中断评测
                print(f"[warn] rerank failed for {record['query']!r}: {exc}")
        rankings.append(dish_names(docs))
        latencies.append((time.perf_counter() - started) * 1000)

    summary = evaluate_rankings(records, rankings, k=args.k)
    summary["config"] = {
        "label": args.label or "default",
        "golden": str(args.golden),
        "ranker": ranker,
        "weights": weights if ranker == "weighted" else None,
        "score_threshold": threshold,
        "top_k": args.top_k,
        "rerank": bool(reranker),
        "reranked_queries": reranked,
        "rerank_model": (
            args.rerank_model or config.reranker.model_name if reranker else None
        ),
        "skip_rerank_explicit": bool(args.skip_rerank_explicit),
    }
    summary["latency_ms"] = {
        "avg": round(mean(latencies), 1) if latencies else 0.0,
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "max": round(max(latencies), 1) if latencies else 0.0,
    }
    if args.misses:
        summary["misses"] = summary["misses"][: args.misses]
    else:
        summary.pop("misses", None)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CookHero 检索 golden set 评测")
    parser.add_argument(
        "--golden", default="data/eval/retrieval_golden.jsonl", help="标注集路径"
    )
    parser.add_argument("--k", type=int, default=5, help="Hit@K / Recall@K 的 K")
    parser.add_argument("--top-k", type=int, default=10, help="每次检索返回的候选数")
    parser.add_argument(
        "--ranker", choices=("weighted", "rrf"), help="混合排序方式，默认取配置"
    )
    parser.add_argument("--weights", help="weighted 权重，例如 0.7,0.3")
    parser.add_argument(
        "--threshold", type=float, help="分数阈值，默认 weighted 用配置值、rrf 用 0"
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument(
        "--offset", type=int, default=0, help="跳过前 N 条，用于分桶对比"
    )
    parser.add_argument("--label", default="", help="本次实验标签，写入输出")
    parser.add_argument("--rerank", action="store_true", help="是否叠加远程重排")
    parser.add_argument(
        "--rerank-limit", type=int, default=0, help="只对前 N 条做重排，0 表示全部"
    )
    parser.add_argument(
        "--rerank-model",
        help="覆盖重排模型，例如 BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        "--skip-rerank-explicit",
        action="store_true",
        help="明确菜名/做法查询跳过重排（模拟线上策略）",
    )
    parser.add_argument("--misses", type=int, default=0, help="输出前 N 条未命中样例")
    parser.add_argument("--output", help="把结果写入 JSON 文件")
    args = parser.parse_args(argv)

    summary = asyncio.run(run(args))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        hitk = summary[f"hit_at_{summary['k']}"]
        print(
            f"{summary['config']['label']}: "
            f"Hit@1={summary['hit_at_1']} Hit@{summary['k']}={hitk} "
            f"MRR={summary['mrr']} -> {args.output}"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
