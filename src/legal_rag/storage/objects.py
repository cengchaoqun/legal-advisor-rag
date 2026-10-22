"""对象存储适配：S3 兼容接口，保存原件与解析产物。

设计变更记录（2026-10-21）：MinIO 开源版已归档，官方二进制返回 410、镜像不可拉取，
因此改用 S3 兼容的 SeaweedFS 承载同一接口；应用侧仍使用标准 S3 客户端，
替换为 MinIO 或其他 S3 服务只需改 endpoint/凭据。
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

from minio import Minio
from minio.error import S3Error

from ..settings import Settings


class ObjectStoreError(RuntimeError):
    pass


class ObjectStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Minio | None = None
        self._bucket = settings.env.minio_bucket

    def _client_or_raise(self) -> Minio:
        if self._client is None:
            env = self._settings.env
            if not env.minio_access_key or not env.minio_secret_key:
                raise ObjectStoreError("对象存储凭据未配置")
            self._client = Minio(
                env.minio_endpoint,
                access_key=env.minio_access_key,
                secret_key=env.minio_secret_key,
                secure=env.minio_secure,
            )
        return self._client

    async def ensure_bucket(self) -> None:
        def _run() -> None:
            client = self._client_or_raise()
            if not client.bucket_exists(self._bucket):
                client.make_bucket(self._bucket)

        await asyncio.to_thread(_run)

    async def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        def _run() -> None:
            client = self._client_or_raise()
            client.put_object(self._bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)

        await asyncio.to_thread(_run)

    async def get_bytes(self, key: str) -> bytes:
        def _run() -> bytes:
            client = self._client_or_raise()
            response = None
            try:
                response = client.get_object(self._bucket, key)
                return response.read()
            finally:
                if response is not None:
                    response.close()
                    response.release_conn()

        try:
            return await asyncio.to_thread(_run)
        except S3Error as exc:  # 对象不存在与权限问题都转成稳定异常
            raise ObjectStoreError(f"对象不可读：{exc.code}") from exc

    async def remove(self, key: str) -> None:
        def _run() -> None:
            client = self._client_or_raise()
            client.remove_object(self._bucket, key)

        try:
            await asyncio.to_thread(_run)
        except S3Error as exc:
            raise ObjectStoreError(str(exc)) from exc

    async def health(self) -> dict[str, Any]:
        try:
            await self.ensure_bucket()
            return {"ok": True, "bucket": self._bucket}
        except Exception as exc:  # 就绪检查需要给出具体原因而不抛异常
            return {"ok": False, "error": str(exc)[:120]}
