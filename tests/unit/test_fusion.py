"""融合与截断单元测试（对应 T09、T10 的算法部分）。"""

from __future__ import annotations

import math

from legal_rag.retrieval.fusion import ScoredItem, cliff_filter, weighted_rrf


def test_weighted_rrf_rank_starts_at_one():
    fused = weighted_rrf({"ordinary": ["a", "b"]}, {"ordinary": 1.0}, k=60, rank_start=1)
    assert fused[0].source_id == "a"
    assert math.isclose(fused[0].score, 1 / 61)
    assert math.isclose(fused[1].score, 1 / 62)


def test_same_chunk_accumulates_across_paths_and_counts_once_per_path():
    fused = weighted_rrf(
        {"ordinary": ["a", "a", "b"], "graph": ["a"]},
        {"ordinary": 1.0, "graph": 0.8},
        k=60,
    )
    scores = {item.source_id: item.score for item in fused}
    assert math.isclose(scores["a"], 1 / 61 + 0.8 / 61)
    assert math.isclose(scores["b"], 1 / 62)
    top = fused[0]
    assert top.source_id == "a"
    assert set(top.recall_sources) == {"ordinary", "graph"}


def test_cliff_filter_keeps_before_first_big_gap():
    items = [
        ScoredItem("a", 0.90),
        ScoredItem("b", 0.80),
        ScoredItem("c", 0.40),
        ScoredItem("d", 0.38),
    ]
    kept = cliff_filter(items, max_examine=6, min_keep=2, abs_gap=0.25, rel_gap=0.35)
    assert [item.source_id for item in kept] == ["a", "b"]


def test_cliff_filter_never_below_min_keep_and_ties_stable():
    items = [ScoredItem("b", 0.5), ScoredItem("a", 0.5)]
    kept = cliff_filter(items, min_keep=2, abs_gap=0.25, rel_gap=0.35)
    assert [item.source_id for item in kept] == ["a", "b"]


def test_cliff_filter_without_gap_returns_examined_prefix():
    items = [ScoredItem(str(i), 1.0 - i * 0.05) for i in range(8)]
    kept = cliff_filter(items, max_examine=6, min_keep=2, abs_gap=0.25, rel_gap=0.35)
    assert len(kept) == 6
