"""模型适配：回答模型（程序外微调并部署）与辅助模型必须严格分离。

回答模型：
- 只调用 LLM_BASE_URL/LLM_MODEL 指向的 OpenAI 兼容服务；
- 生成前核对部署清单的目录摘要，记录 generation_model_id 与 generation_artifact_sha256；
- 服务不可用、清单缺失或摘要不匹配时明确失败，禁止静默回退到辅助云模型。

辅助模型（AUX_*）只用于法律要素抽取与 HyDE，不接管最终回答。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

import httpx

from ..settings import Settings


class ModelServiceUnavailable(RuntimeError):
    code = "MODEL_SERVICE_UNAVAILABLE"


class ModelDeploymentUnverified(RuntimeError):
    code = "MODEL_DEPLOYMENT_UNVERIFIED"


class AuxModelUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class DeploymentManifest:
    served_model_name: str
    directory_sha256: str
    base_model: str
    adapter_sha256: str | None
    max_model_len: int
    tokenizer_sha256: str | None
    path: str


def directory_digest(root: Path) -> str:
    """按“相对路径排序 + 逐文件 SHA256”计算目录摘要，不使用手填标签。"""
    digest = hashlib.sha256()
    # 清单记录目录摘要，本身必须排除，否则写入清单后摘要立刻失效。
    files = sorted((path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() != "deployment_manifest.json"), key=lambda path: path.relative_to(root).as_posix())
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_hash.update(block)
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.hexdigest().encode("ascii"))
    return digest.hexdigest()


def load_deployment_manifest(path: str) -> DeploymentManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ModelDeploymentUnverified(f"部署清单不存在：{path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelDeploymentUnverified(f"部署清单不可读：{exc}") from exc
    required = {"served_model_name", "directory_sha256", "base_model", "max_model_len"}
    missing = required - set(raw)
    if missing:
        raise ModelDeploymentUnverified(f"部署清单缺少字段：{sorted(missing)}")
    return DeploymentManifest(
        served_model_name=raw["served_model_name"],
        directory_sha256=raw["directory_sha256"],
        base_model=raw["base_model"],
        adapter_sha256=raw.get("adapter_sha256"),
        max_model_len=int(raw["max_model_len"]),
        tokenizer_sha256=raw.get("tokenizer_sha256"),
        path=str(manifest_path),
    )


class AnswerModelClient:
    """咨询最终答案模型客户端（流式）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._verified: DeploymentManifest | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return self._settings.env.answer_model_configured()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def verify_manifest(self) -> DeploymentManifest:
        """核对清单与实际模型目录；不接受仅有同名字符串的“已接入”。"""
        if self._verified is not None:
            return self._verified
        env = self._settings.env
        if not env.model_deployment_manifest:
            raise ModelDeploymentUnverified("未配置 MODEL_DEPLOYMENT_MANIFEST，无法证明使用的是微调模型")
        manifest = load_deployment_manifest(env.model_deployment_manifest)
        if not __import__("re").fullmatch(r"[a-f0-9]{64}", manifest.directory_sha256):
            raise ModelDeploymentUnverified("部署产物摘要必须是实际64位SHA256")
        if manifest.served_model_name != env.llm_model:
            raise ModelDeploymentUnverified(
                f"清单服务名 {manifest.served_model_name} 与配置 LLM_MODEL {env.llm_model} 不一致"
            )
        model_dir = Path(env.model_deployment_manifest).parent
        if (model_dir / "config.json").exists():
            actual = directory_digest(model_dir)
            if actual != manifest.directory_sha256:
                raise ModelDeploymentUnverified("模型目录摘要与部署清单不一致，拒绝宣称已接入微调模型")
        if env.model_tokenizer_path:
            if not manifest.tokenizer_sha256 or directory_digest(Path(env.model_tokenizer_path)) != manifest.tokenizer_sha256:
                raise ModelDeploymentUnverified("本地tokenizer目录与已部署模型清单不一致")
        self._verified = manifest
        return manifest

    async def available_models(self) -> list[str]:
        env = self._settings.env
        response = await self._http().get(
            f"{env.llm_base_url.rstrip('/')}/models",
            headers=self._headers(),
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        return [item.get("id", "") for item in payload.get("data", [])]

    def _headers(self) -> dict[str, str]:
        key = self._settings.env.llm_api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def check_model_name(self) -> None:
        """GET /v1/models 必须包含配置的模型 ID，不能取列表第一项。"""
        models = await self.available_models()
        if self._settings.env.llm_model not in models:
            raise ModelServiceUnavailable(
                f"推理服务未提供模型 {self._settings.env.llm_model}；当前可用：{models[:5]}"
            )

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        if not self.configured:
            raise ModelServiceUnavailable("回答模型服务未配置")
        self.verify_manifest()
        env = self._settings.env
        payload = {
            "model": env.llm_model,
            "messages": list(messages),
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = f"{env.llm_base_url.rstrip('/')}/chat/completions"
        has_text = False
        try:
            async with self._http().stream(
                "POST", url, json=payload, headers=self._headers(), timeout=timeout_seconds
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:200]
                    raise ModelServiceUnavailable(f"回答模型返回 {response.status_code}：{body}")
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # HTTP 200的流仍可能携带推理引擎错误，不能吞掉后把空回答标成完成。
                    if chunk.get("error"):
                        error = chunk["error"]
                        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                        raise ModelServiceUnavailable(f"回答模型流返回错误：{str(message)[:200]}")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        has_text = has_text or bool(content.strip())
                        yield content
            if not has_text:
                raise ModelServiceUnavailable("回答模型未返回有效文本")
        except (httpx.HTTPError, OSError) as exc:
            raise ModelServiceUnavailable(f"回答模型服务不可用：{exc}") from exc

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
    ) -> str:
        parts: list[str] = []
        async for piece in self.stream(
            messages, max_tokens=max_tokens, temperature=temperature, timeout_seconds=timeout_seconds
        ):
            parts.append(piece)
        return "".join(parts)

    def artifact_sha256(self) -> str | None:
        """仅在清单已核验时返回产物摘要，避免把猜测写进查询记录。"""
        return self._verified.directory_sha256 if self._verified else None

    async def health(self) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "未配置 LLM_BASE_URL/LLM_MODEL", "configured": False}
        try:
            await self.check_model_name()
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:160], "configured": True}
        try:
            self.verify_manifest()
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:160], "configured": True, "manifest": False}
        return {"ok": True, "configured": True, "manifest": True}


class AuxModelClient:
    """辅助模型：仅做结构化抽取与 HyDE；未配置时返回不可用，由上层走规则或跳过。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return self._settings.env.aux_model_configured()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout_seconds: float = 20.0,
    ) -> str:
        if not self.configured:
            raise AuxModelUnavailable("辅助模型未配置")
        env = self._settings.env
        payload = {
            "model": env.aux_llm_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = await self._http().post(
                f"{env.aux_llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {env.aux_llm_api_key}"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AuxModelUnavailable(f"辅助模型调用失败：{exc}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise AuxModelUnavailable("辅助模型未返回内容")
        return choices[0].get("message", {}).get("content", "")

    async def health(self) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "configured": False, "error": "未配置 AUX_LLM_*"}
        try:
            await self.complete([{"role": "user", "content": "ping"}], max_tokens=8, timeout_seconds=10.0)
            return {"ok": True, "configured": True}
        except Exception as exc:
            return {"ok": False, "configured": True, "error": str(exc)[:160]}


def estimate_tokens(text: str) -> int:
    """保守估算：CJK 字符约 1 token，拉丁字符约 0.3 token。

    仅在没有导出 tokenizer 时使用，并在查询 warnings 中标注估算方式；
    不能替代真实 tokenizer 计数。
    """
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, int(other * 0.3))


class TokenCounter:
    """优先使用导出模型 tokenizer；缺失时退化为带标注的估算。"""

    def __init__(self, tokenizer_path: str | None = None) -> None:
        self._tokenizer = None
        self.mode = "estimate"
        if tokenizer_path and Path(tokenizer_path).exists():
            try:
                from transformers import AutoTokenizer  # 训练组依赖，仅在此处可选导入

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                self.mode = "tokenizer"
            except Exception:
                self._tokenizer = None
                self.mode = "estimate"

    def count_messages(self, messages: Iterable[dict[str, str]]) -> int:
        if self._tokenizer is None:
            total = 0
            for message in messages:
                total += estimate_tokens(message.get("content", "")) + 4
            return total
        texts = [message.get("content", "") for message in messages]
        try:
            encoded = self._tokenizer.apply_chat_template(
                [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            return sum(len(self._tokenizer.encode(text)) for text in texts) + 8 * len(texts)
        return len(encoded)
