"""预算闸门并发与结算（T16）。"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from legal_rag.services.budget import BudgetExceeded, BudgetService
from legal_rag.settings import PricingEntry, PricingTable, Settings

pytestmark = [pytest.mark.usefixtures("require_deps"), pytest.mark.usefixtures("clean_db")]


def paid_settings(runtime) -> Settings:
    config = runtime.settings.config.model_copy(deep=True)
    config.budget.paid_execution_enabled = True
    pricing = PricingTable(
        revision="test-pricing-1",
        vendor="test",
        region="test",
        verified=True,
        effective_date=date(2026, 1, 1),
        checked_at=date(2026, 1, 1),
        currency="CNY",
        entries=[PricingEntry(model="qwen-flash", input_per_1k_yuan=0.001, output_per_1k_yuan=0.002)],
    )
    return Settings(config, runtime.settings.env, pricing)


async def test_paid_calls_disabled_by_default(runtime):
    with pytest.raises(BudgetExceeded) as excinfo:
        await runtime.budget.reserve("req-default", 1.0, model="qwen-flash")
    assert "付费执行默认关闭" in str(excinfo.value)


async def test_concurrent_reserve_only_one_wins(runtime):
    budget = BudgetService(runtime.store, paid_settings(runtime))
    await budget.ensure_account()
    results = await asyncio.gather(
        budget.reserve("req-a", 200.0, model="qwen-flash"),
        budget.reserve("req-b", 200.0, model="qwen-flash"),
        return_exceptions=True,
    )
    successes = [item for item in results if not isinstance(item, BaseException)]
    failures = [item for item in results if isinstance(item, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], BudgetExceeded)
    snapshot = await budget.snapshot()
    assert snapshot.reserved_yuan == 200.0
    assert snapshot.settled_yuan == 0.0


async def test_duplicate_request_id_does_not_double_reserve(runtime):
    budget = BudgetService(runtime.store, paid_settings(runtime))
    await budget.ensure_account()
    await budget.reserve("req-same", 50.0, model="qwen-flash")
    await budget.reserve("req-same", 50.0, model="qwen-flash")
    snapshot = await budget.snapshot()
    assert snapshot.reserved_yuan == 50.0


async def test_settle_moves_reserved_to_settled(runtime):
    budget = BudgetService(runtime.store, paid_settings(runtime))
    await budget.ensure_account()
    await budget.reserve("req-settle", 30.0, model="qwen-flash")
    await budget.settle("req-settle", 12.5, usage={"input_tokens": 1000, "output_tokens": 2000})
    snapshot = await budget.snapshot()
    assert snapshot.reserved_yuan == 0.0
    assert snapshot.settled_yuan == 12.5
