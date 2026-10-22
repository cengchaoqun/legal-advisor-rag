"""联网检索适配：默认关闭；未配置服务时返回 unavailable，绝不生成假网页。

- 只接受配置的 MCP/检索服务返回结果；
- 允许域名来自默认配置（npc.gov.cn/court.gov.cn/mohrss.gov.cn，含子域）；
- 只有拿到正文才标 web（网页原文），仅标题+摘要标 web_snippet（搜索线索）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx

from ..settings import Settings


class WebSearchUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class WebResult:
    title: str
    url: str
    snippet: str
    content: str | None
    publisher: str | None


def domain_allowed(url: str, domains: Sequence[str], allow_subdomains: bool) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    for domain in domains:
        domain = domain.lower()
        if host == domain:
            return True
        if allow_subdomains and host.endswith("." + domain):
            return True
    return False


class WebSearchClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def configured(self) -> bool:
        return self._settings.env.web_configured()

    async def search(self, query: str, *, limit: int, timeout: float) -> list[WebResult]:
        if not self.configured:
            raise WebSearchUnavailable("联网检索服务未配置")
        env = self._settings.env
        headers = {"Authorization": f"Bearer {env.mcp_api_key}"} if env.mcp_api_key else {}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    env.mcp_url, json={"query": query, "limit": limit}, headers=headers
                )
                response.raise_for_status()
                data: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WebSearchUnavailable(f"联网检索失败：{exc}") from exc

        items = data.get("results") if isinstance(data, dict) else data
        results: list[WebResult] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", ""))
            if not domain_allowed(
                url, self._settings.config.sources.source_domains, self._settings.config.sources.allow_subdomains
            ):
                continue
            results.append(
                WebResult(
                    title=str(item.get("title", "")).strip(),
                    url=url,
                    snippet=str(item.get("snippet", "")).strip(),
                    content=(str(item.get("content")).strip() if item.get("content") else None),
                    publisher=item.get("publisher"),
                )
            )
        return results[:limit]

    async def health(self) -> dict[str, object]:
        if not self.configured:
            return {"ok": False, "configured": False}
        try:
            await self.search("劳动法", limit=1, timeout=8.0)
            return {"ok": True, "configured": True}
        except Exception as exc:
            return {"ok": False, "configured": True, "error": str(exc)[:160]}
