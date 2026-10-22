"""确定性切分：优先按法规条款/段落语义边界切分，参数来自默认配置。

同一 version_id 与同一正文必须得到完全相同的 chunk_id 序列，
否则重试会产生重复索引，破坏幂等。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..adapters.parser import ParsedDocument
from ..domain.ids import chunk_id as make_chunk_id
from ..domain.ids import sha256_hex

_ARTICLE_START = re.compile(r"^\s*(第[一二三四五六七八九十百千零〇0-9]+条(?:之[一二三四五六七八九十]+)?)\s*(.*)$")
_HEADING = re.compile(r"^\s*(#{1,6}\s*)?(第[一二三四五六七八九十百千零〇0-9]+[章节]|[一二三四五六七八九十]+、)")
_SENTENCE_END = re.compile(r"(?<=[。；！？!?])")
_SCENARIO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wage_arrears": ("工资", "劳动报酬", "拖欠", "克扣", "欠薪", "支付周期", "加班费"),
    "termination": ("解除", "终止", "辞退", "辞职", "开除", "裁员", "试用期"),
    "economic_compensation": ("经济补偿", "赔偿金", "补偿金", "二倍", "代通知金"),
}


@dataclass(slots=True)
class ChunkDraft:
    ordinal: int
    content: str
    article: str | None
    heading: str | None
    start_char: int
    end_char: int
    page: int | None
    scenario_tags: list[str]


def scenario_tags_for(text: str) -> list[str]:
    tags = [name for name, words in _SCENARIO_KEYWORDS.items() if any(word in text for word in words)]
    return tags or ["other"]


def _split_long(text: str, *, target_min: int, target_max: int, soft_max: int, overlap: int) -> list[str]:
    """按句聚合到目标长度；超长句按硬边界切开，仍保留重叠窗口。"""
    if len(text) <= soft_max:
        return [text]
    sentences = [part for part in _SENTENCE_END.split(text) if part]
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) <= target_max or len(buffer) < target_min:
            buffer += sentence
            continue
        pieces.append(buffer.strip())
        tail = buffer[-overlap:] if overlap else ""
        buffer = tail + sentence
        while len(buffer) > soft_max:
            pieces.append(buffer[:soft_max].strip())
            buffer = buffer[soft_max - overlap :]
    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def _blocks(text: str) -> list[dict[str, object]]:
    """把正文切成“条款/段落”块，并记录全局字符偏移。"""
    blocks: list[dict[str, object]] = []
    heading: str | None = None
    current: dict[str, object] | None = None
    offset = 0
    for line in text.split("\n"):
        line_with_newline = line + "\n"
        if not line.strip():
            offset += len(line_with_newline)
            continue
        heading_match = _HEADING.match(line)
        article_match = _ARTICLE_START.match(line)
        if article_match:
            if current:
                blocks.append(current)
            current = {
                "article": article_match.group(1),
                "heading": heading,
                "start": offset,
                "text": line,
            }
        elif heading_match and not article_match:
            if current:
                blocks.append(current)
                current = None
            heading = line.strip().lstrip("#").strip()
            blocks.append({"article": None, "heading": heading, "start": offset, "text": line})
        elif current is not None:
            current["text"] = f"{current['text']}\n{line}"
        else:
            current = {"article": None, "heading": heading, "start": offset, "text": line}
        offset += len(line_with_newline)
    if current:
        blocks.append(current)
    return blocks


def split_document(
    parsed: ParsedDocument,
    *,
    version_id: str,
    target_min: int,
    target_max: int,
    soft_max: int,
    overlap: int,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    ordinal = 0
    for block in _blocks(parsed.text):
        text = str(block["text"]).strip()
        if not text:
            continue
        article = block.get("article")
        start = int(block["start"])
        pieces = _split_long(text, target_min=target_min, target_max=target_max, soft_max=soft_max, overlap=overlap)
        cursor = start
        for piece in pieces:
            index = parsed.text.find(piece, cursor)
            actual_start = index if index >= 0 else cursor
            actual_end = actual_start + len(piece)
            cursor = actual_end
            drafts.append(
                ChunkDraft(
                    ordinal=ordinal,
                    content=piece,
                    article=str(article) if article else None,
                    heading=str(block.get("heading")) if block.get("heading") else None,
                    start_char=actual_start,
                    end_char=actual_end,
                    page=parsed.page_at(actual_start),
                    scenario_tags=scenario_tags_for(piece),
                )
            )
            ordinal += 1
    return drafts


def to_chunk_document(
    draft: ChunkDraft,
    *,
    version_id: str,
    document_id: str,
    embedding_revision: str,
    quality_flags: list[str],
) -> dict[str, object]:
    return {
        "chunk_id": make_chunk_id(version_id, draft.ordinal, draft.content),
        "document_id": document_id,
        "version_id": version_id,
        "ordinal": draft.ordinal,
        "content": draft.content,
        "location": {
            "page": draft.page,
            "heading": draft.heading,
            "article": draft.article,
            "start_char": draft.start_char,
            "end_char": draft.end_char,
        },
        "scenario_tags": draft.scenario_tags,
        "embedding_revision": embedding_revision,
        "quality_flags": quality_flags,
        "content_sha256": sha256_hex(draft.content),
    }
