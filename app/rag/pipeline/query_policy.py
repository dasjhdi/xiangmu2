"""查询改写预算策略。

目的：能不改写就不改写。首轮问题或本身已经包含明确菜名/做法的查询，
直接拿原问题检索，省掉一次 fast LLM 往返；是否真的需要改写由意图和
指代词决定，这里只做保守判断——拿不准就交给模型。

这些函数都是纯函数，不依赖网络、数据库或模型，便于单测。
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

# 出现这些词说明问题依赖上下文，必须交给改写器做指代消解
_CONTEXT_MARKERS: tuple[str, ...] = (
    "这个",
    "那个",
    "这道",
    "那道",
    "这只",
    "那只",
    "它",
    "它们",
    "上面",
    "前面",
    "刚才",
    "上一道",
    "上一个",
    "再来",
    "换成",
    "改成",
    "还有",
    "继续",
)

# 明确的菜谱/做法问法：即使没有命中菜名目录，也可以安全跳过改写
_EXPLICIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^.{2,30}(怎么做|咋做|如何做|做法|菜谱|配方|做法大全)$"),
    re.compile(r"^.{2,30}(怎么炒|怎么炖|怎么煮|怎么蒸|怎么烤|怎么拌|怎么调|怎么腌)$"),
    re.compile(r"^.{2,30}(需要什么材料|需要哪些材料|用什么食材|放什么调料)$"),
)

_WHITESPACE = re.compile(r"\s+")

# 只有出现这些线索时，才值得花一次 fast LLM 去抽元数据过滤条件。
# 覆盖：难度/时间/饮食约束/过敏原/食材类别/个人文档。
_METADATA_HINTS: tuple[str, ...] = (
    "简单",
    "新手",
    "快速",
    "快手",
    "省事",
    "复杂",
    "进阶",
    "分钟",
    "小时",
    "低脂",
    "减脂",
    "低卡",
    "高蛋白",
    "清淡",
    "不辣",
    "微辣",
    "重辣",
    "素食",
    "全素",
    "荤菜",
    "素菜",
    "汤",
    "主食",
    "甜品",
    "早餐",
    "凉菜",
    "海鲜",
    "水产",
    "猪肉",
    "牛肉",
    "鸡肉",
    "羊肉",
    "过敏",
    "花生",
    "乳制品",
    "麸质",
    "我的",
    "我上传",
    "私人",
)


def normalize_query(text: str) -> str:
    """压缩空白，便于做包含判断。"""
    return _WHITESPACE.sub("", text or "")


def has_context_reference(query: str) -> bool:
    """问题是否依赖对话上下文（指代词、追加条件等）。"""
    text = normalize_query(query)
    return any(marker in text for marker in _CONTEXT_MARKERS)


def is_explicit_recipe_query(
    query: str, dish_names: Iterable[str] = ()
) -> bool:
    """问题是否已经明确到可以直接检索。

    命中菜名目录，或以「XX 怎么做 / XX 的做法」这类明确问法结尾，都算明确。
    """
    text = normalize_query(query)
    if not text or len(text) > 40:
        return False
    if has_context_reference(text):
        return False

    for name in dish_names:
        normalized = normalize_query(str(name))
        if len(normalized) >= 2 and normalized in text:
            return True

    return any(pattern.search(text) for pattern in _EXPLICIT_PATTERNS)


def should_skip_rewrite(
    query: str,
    history_text: str = "",
    dish_names: Iterable[str] = (),
) -> bool:
    """是否跳过 LLM 改写。

    - 没有历史：QueryRewriter 本身就会直接返回，跳过只是把这次机会显式记下来
    - 有历史但问题自洽（含明确菜名/做法、无指代词）：跳过
    - 其余情况一律改写，避免多轮追问丢上下文
    """
    text = normalize_query(query)
    if not text:
        return True
    if not (history_text or "").strip():
        return True
    return is_explicit_recipe_query(text, dish_names)


def dish_names_from_catalog(
    catalog: Mapping[str, Mapping[str, Sequence[str]]] | None,
) -> list[str]:
    """从元数据目录里取出全部菜名，供明确性判断使用。"""
    if not catalog:
        return []
    names: list[str] = []
    for fields in catalog.values():
        if not isinstance(fields, Mapping):
            continue
        values = fields.get("dish_name") or []
        if isinstance(values, str):
            values = [values]
        names.extend(str(value) for value in values if value)
    return names


def needs_metadata_filter(
    query: str, *, has_personal_documents: bool = False
) -> bool:
    """是否需要调用 LLM 抽取元数据过滤条件。

    元数据过滤本身要花一次 fast LLM（实测 2s 左右），而大多数「XX 怎么做」
    根本不需要过滤；没有人命中任何难度/时间/饮食/分类线索时直接跳过。
    有个人文档时保留，因为用户可能用「我的菜谱」这类指代。
    """
    if has_personal_documents:
        return True
    text = normalize_query(query)
    if not text:
        return False
    return any(hint in text for hint in _METADATA_HINTS)


__all__ = [
    "normalize_query",
    "has_context_reference",
    "is_explicit_recipe_query",
    "should_skip_rewrite",
    "dish_names_from_catalog",
    "needs_metadata_filter",
]