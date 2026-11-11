# 检索质量基线与调参记录

> 最后更新：2026-11-02
> 目的：把「检索慢/答不准」从感觉变成可复现的数字。所有结论都可以用下面的命令重跑。

## 一、怎么跑

```bash
# 当前配置（top10，不重排）
python -m scripts.eval_retrieval

# 对比 hybrid 权重
python -m scripts.eval_retrieval --weights 0.7,0.3 --label w07

# RRF
python -m scripts.eval_retrieval --ranker rrf --label rrf

# 叠加远程重排（默认配置里已关闭）
python -m scripts.eval_retrieval --rerank --rerank-limit 50

# 只跑某一桶（exact 0-79 / paraphrase 80-114 / category 115-129）
python -m scripts.eval_retrieval --offset 80 --limit 35
```

标注集：`data/eval/retrieval_golden.jsonl`，共 130 条：

| 桶 | 条数 | 说明 |
| --- | --- | --- |
| exact | 80 | `菜名 + 怎么做 / 的做法 / 需要哪些材料` |
| paraphrase | 35 | 用口感、食材、做法描述一道菜 |
| category | 15 | 菜系、场景、器具、食材类别，允许多个正确答案 |

指标：Hit@1 / Hit@5 / Recall@5 / MRR（多标签查询用 Recall，单标签看 Hit/MRR）。

## 二、基线（2026-11-02，top10，dense 0.8 / BM25 0.2）

| 配置 | Hit@1 | Hit@5 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| **baseline 0.8/0.2** | **87.69%** | **97.69%** | 89.94% | **0.9137** |
| 0.7/0.3 | 87.69% | 97.69% | 89.94% | 0.9137 |
| 0.6/0.4 | 87.69% | 97.69% | 89.94% | 0.9133 |
| 0.5/0.5 | 87.69% | 97.69% | 89.94% | 0.9133 |
| RRF | 87.69% | 97.69% | 89.94% | 0.9133 |
| 阈值 0.25 / 0.30 | 87.69% | 97.69% | 89.94% | 0.9137 |

分桶：

| 桶 | Hit@1 | Hit@5 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| exact | 96.25% | 98.75% | 98.75% | 0.9750 |
| paraphrase | 85.71% | 97.14% | 92.86% | 0.8938 |
| category | 46.67% | 93.33% | 36.17% | 0.6333 |

**结论**

1. 检索不是当前准确率瓶颈：精确菜名 96% 一次命中，改写型 86% 一次命中。
2. hybrid 权重在 0.5–0.8 区间对结果几乎无影响，不必为了「调权重」重建索引；保持 0.8/0.2。
3. 阈值 0.2→0.3 对 Top10 无影响；离题问题（天气/写代码）在 0.2–0.35 下仍能拿到 5–9 篇，说明
   weighted 分数**不是校准过的相似度**，不能靠抬高阈值做「不知道就拒答」。诚实降级应依赖重排分或生成侧。
4. 最弱的是 category（短概念查询，如「川菜」）：纯向量检索只有 46.7% 一次命中，需要改写/查询扩展，
   不是换 embedding 能解决的。

## 三、重排 A/B（Qwen3-Reranker-8B）

| 数据 | 无重排 | 有重排 | 变化 |
| --- | --- | --- | --- |
| exact 前 50 条 | Hit@1 96% / MRR 0.970 | Hit@1 92% / MRR 0.950 | **-4pp / -0.020** |
| paraphrase 35 条 | Hit@1 85.7% / Hit@5 97.1% / MRR 0.894 | 85.7% / 94.3% / 0.900 | Hit@5 **-2.9pp**，MRR +0.006 |
| paraphrase 换 bge-reranker-v2-m3 | — | Hit@1 77.1% / Hit@5 91.4% / MRR 0.838 | 更差 |

**结论**：在真实菜谱语料上，Qwen3-Reranker-8B 对明确菜名会把正确结果往后挪，对改写查询也只有
边际收益；同时它是主要长尾（单次 1–6s）。因此 `config.yml` 默认 `reranker.enabled: false`，
并保留 `skip_rerank_when_explicit` 与超时/候选上限，需要时可直接打开。

## 四、没有做的事（以及原因）

| 方案 | 决定 | 原因 |
| --- | --- | --- |
| 切块加 600 字上限 + 80 重叠 | 暂不做 | 当前 Hit@5 97.7%，重建索引风险大于收益；等 category 桶提上来再评估 |
| `bge-small-zh` → `bge-base-zh` / `bge-m3` | 暂不做 | 需要重建菜谱与个人文档索引、换集合；检索仅 11ms，瓶颈不在这里 |
| 阈值 0.2 → 0.25/0.3 | 不做 | 实测指标无变化，且挡不住离题问题 |
| 重排常开 | 关闭 | 见第三节 |

## 五、延迟预算（P0）

`RAGService.retrieve()` 按阶段打点，后台「链路指标」页可直接看 P50/P95：

| 阶段 | 预算 | 超时行为 |
| --- | --- | --- |
| rewrite | `retrieval.rewrite_timeout_seconds: 2s` | 用原问题检索，状态记 `timeout` |
| metadata | `retrieval.metadata_timeout_seconds: 2s` | 不加过滤条件，状态记 `timeout` |
| search | 无（实测 20–70ms） | — |
| rerank | `reranker.timeout_seconds: 6s`（默认关闭） | 保留检索顺序，只截断 top_n |
| parent_fetch | 无（DB 主键查询） | — |

额外的两个「不调用模型」优化：

- 明确菜名/做法（如「红烧肉怎么做」）跳过改写，直接检索；
- 元数据过滤先用确定性规则匹配难度/分类（`app/rag/pipeline/metadata_policy.py`），
  规则命中就不调 fast LLM；没有难度/时间/饮食/分类线索时也直接跳过。

实测（同一台机器，缓存清空后）：

| 查询 | 优化前 | 优化后 |
| --- | --- | --- |
| 红烧肉怎么做 | 2.9s | **0.04–0.28s** |
| 适合冬天喝的汤 | 5.0s | **0.07s** |
| 再来一道下饭的菜（无历史） | 5.0–8.0s | **0.07s** |
