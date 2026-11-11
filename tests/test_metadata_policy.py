"""元数据确定性规则测试（离线）。"""

from app.rag.pipeline.metadata_policy import build_metadata_expression

CATALOG = {
    "Global Recipes": {
        "difficulty": ["中等", "困难", "简单", "非常困难", "非常简单"],
        "category": ["主食", "早餐", "水产", "汤品", "素菜", "荤菜", "甜品", "饮品"],
    }
}


def test_no_signal_returns_none():
    assert build_metadata_expression("红烧肉怎么做", CATALOG) is None
    assert build_metadata_expression("", CATALOG) is None
    assert build_metadata_expression("来点下饭的菜", CATALOG) is None


def test_missing_catalog_returns_none():
    assert build_metadata_expression("适合冬天喝的汤", None) is None
    assert build_metadata_expression("简单的菜", {}) is None


def test_difficulty_rules_prefer_most_specific():
    assert build_metadata_expression("简单点的红烧肉", CATALOG) == 'difficulty == "简单"'
    assert (
        build_metadata_expression("零基础也能做的菜", CATALOG)
        == 'difficulty == "非常简单"'
    )
    assert (
        build_metadata_expression("有没有复杂一点的硬菜", CATALOG)
        == 'difficulty == "困难"'
    )


def test_category_rules():
    assert (
        build_metadata_expression("适合冬天喝的汤", CATALOG)
        == 'category == "汤品"'
    )
    assert build_metadata_expression("我想吃素菜", CATALOG) == 'category == "素菜"'
    assert build_metadata_expression("海鲜类", CATALOG) == 'category == "水产"'


def test_dish_name_blocks_category_rule():
    assert (
        build_metadata_expression("汤面怎么做", CATALOG, dish_names=["汤面"])
        is None
    )
    # 菜名不影响难度规则
    assert (
        build_metadata_expression("汤面简单吗", CATALOG, dish_names=["汤面"])
        == 'difficulty == "简单"'
    )


def test_difficulty_and_category_can_combine():
    assert (
        build_metadata_expression("简单又清淡的素菜", CATALOG)
        == 'difficulty == "简单" and category == "素菜"'
    )


def test_unknown_catalog_values_are_ignored():
    catalog = {"Global Recipes": {"difficulty": ["中等"], "category": ["其他"]}}
    assert build_metadata_expression("简单的素菜", catalog) is None
