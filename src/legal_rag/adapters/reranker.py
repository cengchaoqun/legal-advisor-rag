"""重排序适配：失败或未配置时保留本地 RRF 顺序并标注降级，不伪装成已完成精排。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import httpx

from ..settings import Settings


class RerankUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class RerankHit:
    index: int
    score: float


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def configured(self) -> bool:
        return self._settings.env.rerank_configured()

    async def rerank(self, query: str, documents: Sequence[str], *, top_n: int, timeout: float) -> list[RerankHit]:
        if not self.configured:
            raise RerankUnavailable("重排序服务未配置")
        env = self._settings.env
        payload = {
            "model": env.rerank_model,
            "input": {"query": query, "documents": list(documents)},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{env.rerank_base_url.rstrip('/')}/rerank",
                    json=payload,
                    headers={"Authorization": f"Bearer {env.rerank_api_key}"},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RerankUnavailable(f"重排序失败：{exc}") from exc

        raw = (data.get("output") or {}).get("results") or data.get("results") or []
        hits: list[RerankHit] = []
        for item in raw:
            try:
                hits.append(RerankHit(index=int(item["index"]), score=float(item["relevance_score"])))
            except (KeyError, TypeError, ValueError):
                continue
        if not hits:
            raise RerankUnavailable("重排序返回空结果")
        return hits

    async def health(self) -> dict[str, object]:
        if not self.configured:
            return {"ok": False, "configured": False}
        try:
            await self.rerank("ping", ["a", "b"], top_n=1, timeout=8.0)
            return {"ok": True, "configured": True}
        except Exception as exc:
            return {"ok": False, "configured": True, "error": str(exc)[:160]}
