"""元数据过滤的确定性规则。

元数据枚举很小（难度 6 个值、分类 13 个值），用一次 fast LLM 去抽
`difficulty == "简单"` 这类表达式既慢（实测 1.8–15s）又容易抽空。
这里先用同义词规则直接命中，命中不了再交给 LLM。

纯函数，方便单测。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from app.rag.pipeline.query_policy import normalize_query

# 长词在前，避免「非常简单」先被「简单」吃掉
_DIFFICULTY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("非常困难", ("非常困难", "特别难", "很难")),
    ("困难", ("困难", "复杂", "进阶", "有挑战")),
    ("中等", ("中等", "普通难度", "一般难度")),
    ("非常简单", ("非常简单", "零基础", "新手友好")),
    ("简单", ("简单", "新手", "容易", "快手", "快速", "省事")),
)

# 分类词容易和菜名撞车（如「汤面」），命中菜名时不套用分类规则
_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("汤品", ("汤品", "煲汤", "炖汤", "喝的汤", "例汤")),
    ("素菜", ("素菜", "素食", "全素", "纯素")),
    ("荤菜", ("荤菜", "肉菜")),
    ("早餐", ("早餐", "早饭")),
    ("甜品", ("甜品", "甜点")),
    ("饮品", ("饮品", "饮料")),
    ("主食", ("主食", "面食")),
    ("水产", ("水产", "海鲜")),
    ("调料", ("调料", "酱料")),
)


def _known(value: str, catalog_values: Iterable[str]) -> bool:
    return value in set(catalog_values)


def build_metadata_expression(
    query: str,
    catalog: Mapping[str, Mapping[str, Sequence[str]]] | None,
    *,
    dish_names: Iterable[str] = (),
) -> str | None:
    """从查询中直接生成 Milvus 过滤表达式。

    只使用全局菜谱目录（个人文档仍走原有 user_id 过滤）。
    无法确定时返回 None，由调用方决定是否交给 LLM。
    """
    if not catalog:
        return None
    global_fields = catalog.get("Global Recipes") or {}
    difficulties = global_fields.get("difficulty") or []
    categories = global_fields.get("category") or []
    text = normalize_query(query)
    if not text:
        return None

    conditions: list[str] = []

    for canonical, synonyms in _DIFFICULTY_RULES:
        if _known(canonical, difficulties) and any(
            word in text for word in synonyms
        ):
            conditions.append(f'difficulty == "{canonical}"')
            break  # 只取最具体的一档

    # 菜名里可能带「汤」「面」等字，命中菜名时不套分类规则
    has_dish_name = any(
        name and normalize_query(name) in text for name in dish_names
    )
    if not has_dish_name:
        for canonical, synonyms in _CATEGORY_RULES:
            if _known(canonical, categories) and any(
                word in text for word in synonyms
            ):
                conditions.append(f'category == "{canonical}"')
                break  # 只取一个分类，避免 AND 出空集

    return " and ".join(conditions) if conditions else None


__all__ = ["build_metadata_expression"]
