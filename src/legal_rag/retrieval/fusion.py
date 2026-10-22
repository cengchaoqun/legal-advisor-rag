"""融合与截断：加权 RRF 与断崖过滤，纯函数便于单测。

契约：
- score(d) = Σ weight / (k + rank(d))，rank 从 1 开始；
- 同一路内同一 source_id 只计一次，不同路径命中可累加并保留 recall_sources；
- 断崖：考察前 6 条，从第 2 条之后找首个 abs>=0.25 且 rel>=0.35 的相邻差，至少保留 2 条。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(slots=True)
class FusedItem:
    source_id: str
    score: float = 0.0
    recall_sources: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)


def weighted_rrf(
    ranked_lists: Mapping[str, Sequence[str]],
    weights: Mapping[str, float],
    *,
    k: int = 60,
    rank_start: int = 1,
) -> list[FusedItem]:
    items: dict[str, FusedItem] = {}
    for path, ordered in ranked_lists.items():
        weight = float(weights.get(path, 1.0))
        seen_in_path: set[str] = set()
        rank = rank_start
        for source_id in ordered:
            if source_id in seen_in_path:
                continue
            seen_in_path.add(source_id)
            item = items.setdefault(source_id, FusedItem(source_id=source_id))
            item.score += weight / (k + rank)
            item.ranks[path] = rank
            if path not in item.recall_sources:
                item.recall_sources.append(path)
            rank += 1
    return sorted(items.values(), key=lambda item: (-item.score, item.source_id))


def cliff_filter(
    items: Sequence["ScoredItem"],
    *,
    max_examine: int = 6,
    min_keep: int = 2,
    abs_gap: float = 0.25,
    rel_gap: float = 0.35,
) -> list["ScoredItem"]:
    """按精排分数截断；分数相同按 source_id 升序，避免顺序不稳定。"""
    ordered = sorted(items, key=lambda item: (-item.score, item.source_id))[:max_examine]
    if len(ordered) <= min_keep:
        return ordered
    for index in range(1, len(ordered) - 1):
        current = ordered[index].score
        following = ordered[index + 1].score
        gap = current - following
        relative = gap / max(abs(current), 0.00000001)
        if gap >= abs_gap and relative >= rel_gap:
            return ordered[: index + 1]
    return ordered


@dataclass(slots=True)
class ScoredItem:
    source_id: str
    score: float


def dedupe_by_source(items: Iterable[FusedItem]) -> list[FusedItem]:
    seen: set[str] = set()
    result: list[FusedItem] = []
    for item in items:
        if item.source_id in seen:
            continue
        seen.add(item.source_id)
        result.append(item)
    return result


def web_source_id(url: str, text: str) -> str:
    from ..domain.ids import web_source_id as make

    return make(url, make_hash(text))


def make_hash(text: str) -> str:
    from ..domain.ids import sha256_hex

    return sha256_hex(text)
