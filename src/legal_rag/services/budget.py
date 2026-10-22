"""预算闸门：单文档原子预留/结算，防止并发检查后各自超额放行。

- settled + reserved + estimated_unsettled 不得超过 300 元；
- 同一 request_id 不重复预留/结算；
- 未验证价格或 paid_execution_enabled=false 时拒绝付费调用；
- 自托管回答模型按资源时间计费，不走 token 预留（serving.billing=resource_time）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymongo import ReturnDocument

from ..settings import ConfigError, Settings
from ..storage.mongo import MongoStore, utcnow


class BudgetExceeded(RuntimeError):
    code = "BUDGET_LIMIT"


class PricingMissing(RuntimeError):
    code = "PRICING_NOT_CONFIGURED"


@dataclass(slots=True)
class Reservation:
    request_id: str
    amount: float


class BudgetService:
    def __init__(self, store: MongoStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings

    async def ensure_account(self) -> dict[str, Any]:
        now = utcnow()
        await self._store.collection("budget_account").update_one(
            {"_id": "project"},
            {
                "$setOnInsert": {
                    "limit": self._settings.config.budget.total_limit,
                    "settled": 0.0,
                    "reserved": 0.0,
                    "estimated_unsettled": 0.0,
                    "api_spent": 0.0,
                    "gpu_spent": 0.0,
                    "reservations": {},
                    "pricing_revision": self._settings.pricing.revision if self._settings.pricing else "unconfigured",
                    "last_reconciled_at": None,
                    "created_at": now,
                }
            },
            upsert=True,
        )
        account = await self._store.get_budget_account()
        return account or {}

    def _require_paid_enabled(self) -> None:
        if not self._settings.config.budget.paid_execution_enabled:
            raise BudgetExceeded("付费执行默认关闭（budget.paid_execution_enabled=false），拒绝发起付费调用")
        if self._settings.pricing is None:
            raise PricingMissing("价格配置缺失或未验证，拒绝付费调用")

    async def reserve(self, request_id: str, amount: float, *, model: str, kind: str = "api") -> Reservation:
        """原子预留：$expr 同时校验总额，reservations 键保证幂等。"""
        self._require_paid_enabled()
        if amount < 0:
            raise ValueError("预留金额不能为负")
        await self.ensure_account()
        now = utcnow()
        key = f"reservations.{request_id.replace('.', '_')}"
        result = await self._store.collection("budget_account").update_one(
            {
                "_id": "project",
                key: {"$exists": False},
                "$expr": {
                    "$lte": [
                        {"$add": ["$settled", "$reserved", "$estimated_unsettled", amount]},
                        "$limit",
                    ]
                },
            },
            {
                "$inc": {"reserved": amount},
                "$set": {
                    key: {
                        "amount": amount,
                        "model": model,
                        "kind": kind,
                        "status": "reserved",
                        "created_at": now,
                    }
                },
            },
        )
        if result.modified_count != 1:
            account = await self.ensure_account()
            if key.split(".", 1)[1] in (account.get("reservations") or {}):
                # 幂等：同一 request_id 已预留，直接复用
                return Reservation(request_id, amount)
            raise BudgetExceeded(
                f"预算不足：已结算 {account.get('settled', 0):.2f} + 已预留 {account.get('reserved', 0):.2f}"
                f" + 未结算估算 {account.get('estimated_unsettled', 0):.2f}，上限 {account.get('limit')}"
            )
        return Reservation(request_id, amount)

    async def settle(self, request_id: str, actual_amount: float, *, usage: dict[str, Any] | None = None) -> None:
        """按实际用量结算：reserved 退回、settled 增加，写成本账本。"""
        await self.ensure_account()
        key = f"reservations.{request_id.replace('.', '_')}"
        account = await self._store.collection("budget_account").find_one_and_update(
            {"_id": "project", key: {"$exists": True}, f"{key}.status": "reserved"},
            {
                "$inc": {"reserved": -1 * (await self._reserved_amount(key)), "settled": actual_amount},
                "$set": {f"{key}.status": "settled", f"{key}.settled_amount": actual_amount},
            },
            return_document=ReturnDocument.AFTER,
        )
        await self._store.insert_cost_entry(
            {
                "entry_id": str(uuid.uuid4()),
                "request_id": request_id,
                "kind": "api",
                "amount_yuan": round(actual_amount, 6),
                "usage": usage or {},
                "pricing_revision": self._settings.pricing.revision if self._settings.pricing else "unconfigured",
                "created_at": utcnow(),
            }
        )

    async def _reserved_amount(self, key: str) -> float:
        account = await self._store.get_budget_account() or {}
        reservations = account.get("reservations") or {}
        entry = reservations.get(key.split(".", 1)[1], {})
        return float(entry.get("amount", 0.0))

    async def release(self, request_id: str) -> None:
        """调用失败时释放预留，避免预算被永久占用。"""
        await self.ensure_account()
        key = f"reservations.{request_id.replace('.', '_')}"
        amount = await self._reserved_amount(key)
        if amount <= 0:
            return
        await self._store.collection("budget_account").update_one(
            {"_id": "project", key: {"$exists": True}, f"{key}.status": "reserved"},
            {
                "$inc": {"reserved": -amount},
                "$set": {f"{key}.status": "released", f"{key}.released_at": utcnow()},
            },
        )

    async def snapshot(self):
        account = await self.ensure_account()
        from ..domain.models import Cost

        return Cost(
            total_limit_yuan=300,
            settled_yuan=round(float(account.get("settled", 0.0)), 4),
            reserved_yuan=round(float(account.get("reserved", 0.0)), 4),
            estimated_unsettled_yuan=round(float(account.get("estimated_unsettled", 0.0)), 4),
            pricing_revision=str(
                account.get("pricing_revision")
                or (self._settings.pricing.revision if self._settings.pricing else "unconfigured")
            ),
            last_reconciled_at=account.get("last_reconciled_at"),
        )

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if self._settings.pricing is None:
            raise PricingMissing("价格配置缺失或未验证")
        entry = self._settings.pricing.price_for(model)
        if entry is None:
            raise ConfigError(f"模型 {model} 缺少有效单价")
        return (input_tokens / 1000.0) * entry.input_per_1k_yuan + (
            output_tokens / 1000.0
        ) * entry.output_per_1k_yuan
