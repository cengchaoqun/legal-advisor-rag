"""文档解析：文本型 PDF 与 Markdown。

- PDF 使用 pypdf 逐页提取，保留页码；提取不到有效文字时返回 PARSER_UNSUPPORTED，
  绝不生成“空正文成功”；
- Markdown 保留标题层级与条号，去掉纯排版标记；
- MinerU 作为可选增强解析器保留接口，未配置时与扫描件同样处理。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ParserError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class PageText:
    page: int
    text: str


@dataclass(slots=True)
class ParsedDocument:
    text: str
    pages: list[PageText] = field(default_factory=list)
    partial: bool = False
    parser: str = "pypdf_text"

    def page_at(self, char_offset: int) -> int | None:
        """根据字符偏移估算页码：PDF 分页边界由逐页文本累积得到。"""
        if not self.pages:
            return None
        cursor = 0
        for page in self.pages:
            cursor += len(page.text) + 1
            if char_offset < cursor:
                return page.page
        return self.pages[-1].page


_ARTICLE_RE = re.compile(r"^\s*第[一二三四五六七八九十百千零〇0-9]+条", re.MULTILINE)


def looks_like_legal_text(text: str) -> bool:
    return len(_ARTICLE_RE.findall(text)) >= 2


def parse_pdf(data: bytes, *, max_pages: int) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError) as exc:
        raise ParserError("PARSER_UNSUPPORTED", f"PDF 无法解析：{exc}") from exc

    if len(reader.pages) > max_pages:
        # 异步阶段发现超页：写 TaskStatus.error，不在 202 之后补发 HTTP 413
        raise ParserError("PAGE_LIMIT_EXCEEDED", f"PDF 页数 {len(reader.pages)} 超过上限 {max_pages}")

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # 单页异常不应让整份资料静默成功
            text = ""
        normalized = text.replace("\r\n", "\n").strip()
        pages.append(PageText(page=index, text=normalized))

    total_chars = sum(len(page.text) for page in pages)
    # 每页平均不足 30 字视为无可用文本层（扫描件），明确不支持而非空成功
    if total_chars == 0 or total_chars / max(1, len(pages)) < 30:
        raise ParserError(
            "PARSER_UNSUPPORTED",
            "PDF 未提取到有效文本层，疑似扫描件；首期不支持 OCR，请提供文本型 PDF 或 Markdown",
        )

    partial = any(not page.text for page in pages)
    text = "\n".join(page.text for page in pages)
    return ParsedDocument(text=text, pages=pages, partial=partial, parser="pypdf_text")


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")


def parse_markdown(data: bytes) -> ParsedDocument:
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = data.decode("utf-8", errors="replace")
    text = raw.replace("\r\n", "\n")
    # 仅去掉链接目标与强调标记，保留标题符号与条号，便于按条切分
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPHASIS.sub("", text)
    if not text.strip():
        raise ParserError("PARSER_UNSUPPORTED", "Markdown 内容为空")
    return ParsedDocument(text=text, pages=[], partial=False, parser="markdown_text")


def parse(filename: str, data: bytes, *, max_pages: int) -> ParsedDocument:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "pdf":
        return parse_pdf(data, max_pages=max_pages)
    if suffix in {"md", "markdown"}:
        return parse_markdown(data)
    raise ParserError("INVALID_FILE_TYPE", "仅支持文本型 PDF 与 Markdown")


def detect_mime(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {"pdf": "application/pdf", "md": "text/markdown", "markdown": "text/markdown"}.get(
        suffix, "application/octet-stream"
    )
