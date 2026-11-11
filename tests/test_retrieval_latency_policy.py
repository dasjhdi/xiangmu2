"""P0 检索延迟策略单测：改写预算/跳过、元数据超时、重排候选与超时。"""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from app.observability import semantic_metrics
from app.rag.pipeline.query_policy import (
    dish_names_from_catalog,
    has_context_reference,
    is_explicit_recipe_query,
    needs_metadata_filter,
    should_skip_rewrite,
)


def _load_rag_service_module():
    """直接加载 rag_service.py，避免 app.services 包的初始化副作用。"""
    module_name = "_rag_service_latency_under_test"
    if module_name not in sys.modules:
        path = Path("app/services/rag_service.py")
        source = path.read_text(encoding="utf-8").replace(
            "\nrag_service_instance = RAGService()",
            "\nrag_service_instance = None",
        )
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        sys.modules[module_name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
    return sys.modules[module_name]


def _retrieval_config(**overrides):
    values = dict(
        top_k=9,
        multi_query_enabled=False,
        max_queries=3,
        rewrite_timeout_seconds=0.05,
        skip_rewrite_when_explicit=True,
        metadata_timeout_seconds=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_service(monkeypatch, *, retrieval=None, reranker=None, docs=None):
    module = _load_rag_service_module()
    service = module.RAGService.__new__(module.RAGService)
    service.retrieval_modules = {"recipes": object()}
    service.config = SimpleNamespace(
        retrieval=retrieval or _retrieval_config(),
        reranker=reranker or SimpleNamespace(enabled=False),
    )
    service.reranker = None

    async def _identity(documents):
        return documents

    monkeypatch.setattr(
        module,
        "document_processor",
        SimpleNamespace(post_process_retrieval=_identity),
    )

    async def _execute(*args, **kwargs):
        return list(docs or [])

    service._execute_retrieval = _execute

    async def _no_filter(*args, **kwargs):
        return None

    service.metadata_filter_extractor = SimpleNamespace(
        build_filter_expression=_no_filter
    )
    service.generation_module = SimpleNamespace(
        rewrite_query=lambda *a, **k: None
    )
    return service


def _doc(name: str) -> Document:
    return Document(page_content=f"{name} 的做法", metadata={"dish_name": name})


# ---------------------------------------------------------------------------
# 纯策略
# ---------------------------------------------------------------------------


def test_explicit_recipe_query_detection():
    assert is_explicit_recipe_query("红烧肉怎么做")
    assert is_explicit_recipe_query("番茄炒蛋的做法")
    assert is_explicit_recipe_query("宫保鸡丁需要什么材料")
    assert is_explicit_recipe_query("随便炒个菜", ["随便炒个菜"])
    assert not is_explicit_recipe_query("这个怎么做")
    assert not is_explicit_recipe_query("来点清淡的")


def test_context_reference_detection():
    assert has_context_reference("这个需要煮多久")
    assert has_context_reference("换成牛肉呢")
    assert not has_context_reference("红烧肉怎么做")


def test_should_skip_rewrite_rules():
    # 首轮直接跳过改写：QueryRewriter 本身也会直接返回
    assert should_skip_rewrite("来点清淡的", history_text="")
    # 多轮里问题自洽才跳过
    assert should_skip_rewrite("红烧肉怎么做", history_text="用户：上次聊了鱼")
    # 多轮里含指代必须改写
    assert not should_skip_rewrite("这个怎么做", history_text="用户：红烧肉")
    # 多轮里模糊需求必须改写
    assert not should_skip_rewrite("来点清淡的", history_text="用户：红烧肉")


def test_dish_names_from_catalog():
    catalog = {
        "Global Recipes": {"dish_name": ["红烧肉", "番茄炒蛋"], "category": ["荤菜"]},
        "Personal Documents": {"dish_name": ["我的拿手菜"]},
    }
    assert dish_names_from_catalog(catalog) == ["红烧肉", "番茄炒蛋", "我的拿手菜"]
    assert dish_names_from_catalog(None) == []


def test_needs_metadata_filter_only_for_filter_signals():
    assert not needs_metadata_filter("红烧肉怎么做")
    assert not needs_metadata_filter("再来一道下饭的菜")
    assert needs_metadata_filter("红烧肉怎么做简单点")
    assert needs_metadata_filter("适合冬天喝的汤")
    assert needs_metadata_filter("有没有低脂的菜")
    assert needs_metadata_filter("我上传的菜谱里有什么")
    assert needs_metadata_filter("随便看看", has_personal_documents=True)


# ---------------------------------------------------------------------------
# RAGService 预算控制
# ---------------------------------------------------------------------------


async def test_rewrite_skipped_for_self_contained_query(monkeypatch):
    semantic_metrics.reset()
    service = _build_service(monkeypatch, docs=[_doc("红烧肉")])

    async def _must_not_call(*args, **kwargs):
        raise AssertionError("explicit query should skip rewrite")

    service.generation_module = SimpleNamespace(rewrite_query=_must_not_call)

    result = await service.retrieve("红烧肉怎么做", history_text="用户：上次聊了鱼")

    assert result.rewritten_query == "红烧肉怎么做"
    stage = semantic_metrics.snapshot()["pipeline"]["stages"]["rewrite"]
    assert stage["status"] == {"skipped": 1}
    semantic_metrics.reset()


async def test_rewrite_timeout_falls_back_to_original(monkeypatch):
    semantic_metrics.reset()
    service = _build_service(monkeypatch, docs=[_doc("红烧肉")])
    called = 0

    async def _slow_rewrite(*args, **kwargs):
        nonlocal called
        called += 1
        await asyncio.sleep(1)
        return "should not be used"

    service.generation_module = SimpleNamespace(rewrite_query=_slow_rewrite)

    started = asyncio.get_running_loop().time()
    result = await service.retrieve("换成别的做法", history_text="用户：红烧肉")
    elapsed = asyncio.get_running_loop().time() - started

    assert called == 1
    assert result.rewritten_query == "换成别的做法"
    assert elapsed < 0.5
    stage = semantic_metrics.snapshot()["pipeline"]["stages"]["rewrite"]
    assert stage["status"] == {"timeout": 1}
    semantic_metrics.reset()


async def test_metadata_timeout_degrades_without_filter(monkeypatch):
    semantic_metrics.reset()
    service = _build_service(monkeypatch, docs=[_doc("红烧肉")])

    async def _slow_filter(*args, **kwargs):
        await asyncio.sleep(1)
        return 'dish_name == "红烧肉"'

    service.metadata_filter_extractor = SimpleNamespace(
        build_filter_expression=_slow_filter
    )

    started = asyncio.get_running_loop().time()
    result = await service.retrieve("红烧肉需要多少分钟", history_text="用户：上次聊了鱼")
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.5
    assert len(result.documents) == 1
    stage = semantic_metrics.snapshot()["pipeline"]["stages"]["metadata"]
    assert stage["status"] == {"timeout": 1}
    semantic_metrics.reset()


async def test_rerank_candidates_are_capped(monkeypatch):
    semantic_metrics.reset()
    seen: list[int] = []

    class _EchoReranker:
        fallback_used = 0

        async def rerank(self, query, documents):
            seen.append(len(documents))
            return list(documents)

    service = _build_service(
        monkeypatch,
        retrieval=_retrieval_config(skip_rewrite_when_explicit=False),
        docs=[_doc("红烧肉"), _doc("鱼香茄子"), _doc("番茄炒蛋")],
        reranker=SimpleNamespace(
            enabled=True,
            max_candidates=2,
            top_n=1,
            timeout_seconds=1.0,
            skip_rerank_when_explicit=False,
        ),
    )
    service.reranker = _EchoReranker()

    result = await service.retrieve("红烧肉怎么做", history_text="用户：上次聊了鱼")

    assert seen == [2]
    assert len(result.documents) == 1
    stage = semantic_metrics.snapshot()["pipeline"]["stages"]["rerank"]
    assert stage["status"] == {"ok": 1}
    semantic_metrics.reset()


async def test_rerank_timeout_keeps_retrieval_order(monkeypatch):
    semantic_metrics.reset()

    class _SlowReranker:
        fallback_used = 0

        async def rerank(self, query, documents):
            await asyncio.sleep(1)
            return []

    service = _build_service(
        monkeypatch,
        retrieval=_retrieval_config(skip_rewrite_when_explicit=False),
        docs=[_doc("红烧肉"), _doc("鱼香茄子"), _doc("番茄炒蛋")],
        reranker=SimpleNamespace(
            enabled=True,
            max_candidates=3,
            top_n=2,
            timeout_seconds=0.05,
            skip_rerank_when_explicit=False,
        ),
    )
    service.reranker = _SlowReranker()

    result = await service.retrieve("红烧肉怎么做", history_text="用户：上次聊了鱼")

    assert [doc.metadata["dish_name"] for doc in result.documents] == [
        "红烧肉",
        "鱼香茄子",
    ]
    stage = semantic_metrics.snapshot()["pipeline"]["stages"]["rerank"]
    assert stage["status"] == {"timeout": 1}
    semantic_metrics.reset()


async def test_explicit_query_skips_rerank(monkeypatch):
    semantic_metrics.reset()
    calls = 0

    class _Reranker:
        fallback_used = 0

        async def rerank(self, query, documents):
            nonlocal calls
            calls += 1
            return list(documents)

    service = _build_service(
        monkeypatch,
        docs=[_doc("红烧肉"), _doc("鱼香茄子"), _doc("番茄炒蛋")],
        reranker=SimpleNamespace(
            enabled=True,
            max_candidates=6,
            top_n=2,
            timeout_seconds=1.0,
            skip_rerank_when_explicit=True,
        ),
    )
    service.reranker = _Reranker()

    result = await service.retrieve("红烧肉怎么做", history_text="用户：上次聊了鱼")

    assert calls == 0
    assert len(result.documents) == 2
    stage = semantic_metrics.snapshot()["pipeline"]["stages"]["rerank"]
    assert stage["status"] == {"skipped": 1}
    semantic_metrics.reset()


async def test_stage_total_is_recorded(monkeypatch):
    semantic_metrics.reset()
    service = _build_service(monkeypatch, docs=[_doc("红烧肉")])

    await service.retrieve("红烧肉怎么做", history_text="用户：上次聊了鱼")

    stages = semantic_metrics.snapshot()["pipeline"]["stages"]
    assert {"search", "metadata", "parent_fetch", "total"} <= set(stages)
    assert stages["total"]["samples"] == 1
    assert stages["total"]["max_ms"] >= stages["search"]["max_ms"]
    semantic_metrics.reset()
