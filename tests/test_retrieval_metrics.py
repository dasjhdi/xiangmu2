"""检索基线指标与 golden set 完整性测试（离线）。"""

import json
from pathlib import Path

import pytest

from app.evaluation.retrieval_metrics import (
    evaluate_rankings,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scripts.eval_retrieval import dish_names, load_golden


def test_hit_at_k_and_recall():
    ranked = ["a", "b", "c", "d"]
    relevant = {"c", "d", "z"}

    assert hit_at_k(ranked, relevant, 1) == 0.0
    assert hit_at_k(ranked, relevant, 3) == 1.0
    assert recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, relevant, 0) == 0.0
    assert recall_at_k(ranked, set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"z"}) == 0.0
    assert reciprocal_rank([], {"z"}) == 0.0


def test_evaluate_rankings_overall_and_buckets():
    records = [
        {"query": "q1", "relevant": ["a"], "bucket": "exact"},
        {"query": "q2", "relevant": ["b"], "bucket": "exact"},
        {"query": "q3", "relevant": ["c", "d"], "bucket": "paraphrase"},
    ]
    rankings = [["a", "x"], ["x", "b"], ["x", "c", "d"]]

    summary = evaluate_rankings(records, rankings, k=2)

    assert summary["samples"] == 3
    assert summary["hit_at_1"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["hit_at_2"] == 1.0
    assert summary["recall_at_2"] == pytest.approx((1 + 1 + 0.5) / 3, abs=1e-4)
    assert summary["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 2) / 3, abs=1e-4)
    assert summary["by_bucket"]["exact"]["samples"] == 2
    assert summary["by_bucket"]["exact"]["hit_at_2"] == 1.0
    assert summary["misses"] == []


def test_evaluate_rankings_reports_misses():
    records = [{"query": "q", "relevant": ["a"], "bucket": "exact"}]
    summary = evaluate_rankings(records, [["x", "y"]], k=2)

    assert summary["hit_at_2"] == 0.0
    assert summary["misses"] == [
        {"query": "q", "bucket": "exact", "relevant": ["a"], "ranked": ["x", "y"]}
    ]


def test_evaluate_rankings_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_rankings([{"query": "q", "relevant": ["a"]}], [])


def test_dish_names_dedupes_and_skips_index():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="", metadata={"dish_name": "红烧肉"}),
        Document(page_content="", metadata={"dish_name": "红烧肉"}),
        Document(page_content="", metadata={"dish_name": "菜谱索引"}),
        Document(page_content="", metadata={}),
        Document(page_content="", metadata={"dish_name": "鱼香茄子"}),
    ]

    assert dish_names(docs) == ["红烧肉", "鱼香茄子"]


def test_golden_set_is_valid_and_names_exist():
    path = Path("data/eval/retrieval_golden.jsonl")
    records = load_golden(path)

    assert len(records) == 130
    assert {record["bucket"] for record in records} == {
        "exact",
        "paraphrase",
        "category",
    }

    corpus = {item.stem for item in Path("data/HowToCook/dishes").rglob("*.md")}
    for record in records:
        assert record["relevant"], record["query"]
        assert set(record["relevant"]) <= corpus, record["query"]


def test_golden_set_queries_are_unique():
    records = load_golden(Path("data/eval/retrieval_golden.jsonl"))
    queries = [record["query"] for record in records]
    assert len(queries) == len(set(queries))


def test_load_golden_rejects_bad_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        load_golden(path)


def test_load_golden_normalizes_string_relevant(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text(
        json.dumps({"query": "q", "relevant": "红烧肉"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records = load_golden(path)

    assert records[0]["relevant"] == ["红烧肉"]
    assert records[0]["bucket"] == "default"
