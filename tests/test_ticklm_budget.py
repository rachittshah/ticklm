from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from ticklm_bench.budget import (
    BudgetBlocked,
    BudgetExceeded,
    BudgetLedger,
    IdempotencyConflict,
    Pricing,
    estimate_max_cost,
)


def test_duplicate_reservation_is_idempotent_but_not_executable_twice(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_usd="2")

    first = ledger.reserve("attempt-1", "1.25", {"model": "provider/model"})
    duplicate = ledger.reserve("attempt-1", "1.25", {"model": "provider/model"})

    assert first.created is True
    assert first.can_execute is True
    assert duplicate.created is False
    assert duplicate.can_execute is False
    assert duplicate.state == "reserved"
    assert ledger.summary()["reserved_usd"] == "1.250000"
    assert ledger.summary()["remaining_usd"] == "0.750000"


def test_concurrent_distinct_reservations_never_overspend(tmp_path):
    path = tmp_path / "budget.sqlite"
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(BudgetLedger(path, limit_usd="10").reserve,
                               f"request-{i}", "1") for i in range(24)]
        results = []
        rejected = 0
        for future in futures:
            try:
                results.append(future.result())
            except BudgetExceeded:
                rejected += 1

    assert sum(result.created for result in results) == 10
    assert rejected == 14
    assert BudgetLedger(path, limit_usd="10").summary()["remaining_usd"] == "0.000000"


def test_concurrent_duplicate_reservations_create_one_executable_claim(tmp_path):
    path = tmp_path / "budget.sqlite"
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(BudgetLedger(path, limit_usd="5").reserve,
                               "same-request", "1", {"item": "same"}) for _ in range(16)]
        claims = [future.result() for future in futures]

    assert sum(claim.can_execute for claim in claims) == 1
    assert BudgetLedger(path, limit_usd="5").summary()["reserved_usd"] == "1.000000"


def test_unknown_cost_consumes_the_reservation_until_resolved(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_usd="3")
    ledger.reserve("unknown-cost", "1.4")

    unknown = ledger.settle("unknown-cost")
    assert unknown.unknown is True
    assert unknown.charged_usd == Decimal("1.400000")
    summary = ledger.summary()
    assert summary["reserved_usd"] == "0.000000"
    assert summary["unknown_usd"] == "1.400000"
    assert summary["remaining_usd"] == "1.600000"

    resolved = ledger.settle("unknown-cost", "0.2")
    assert resolved.unknown is False
    assert resolved.charged_usd == Decimal("0.200000")
    assert ledger.summary()["remaining_usd"] == "2.800000"


def test_known_settlement_releases_unused_reservation_and_is_idempotent(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_usd="2")
    ledger.reserve("known-cost", "1")

    first = ledger.settle("known-cost", "0.4")
    second = ledger.settle("known-cost", "0.4")

    assert first.charged_usd == second.charged_usd == Decimal("0.400000")
    assert first.unknown is second.unknown is False
    assert ledger.summary()["remaining_usd"] == "1.600000"
    with pytest.raises(IdempotencyConflict):
        ledger.settle("known-cost", "0.5")


def test_actual_cost_above_reservation_blocks_future_paid_calls(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_usd="5")
    ledger.reserve("overrun", "1")

    settlement = ledger.settle("overrun", "1.2")

    assert settlement.exceeded_reservation is True
    assert settlement.overrun_usd == Decimal("0.200000")
    assert settlement.blocked is True
    assert ledger.summary()["blocked_on_overrun"] is True
    assert ledger.summary()["reservation_overrun_usd"] == "0.200000"
    with pytest.raises(BudgetBlocked):
        ledger.reserve("next", "0.1")


def test_large_provider_overrun_does_not_overflow_aggregate_accounting(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_usd="5")
    ledger.reserve("first", "1")
    ledger.settle("first", "1")
    ledger.reserve("large-overrun", "1")

    ledger.settle("large-overrun", "9223372036854.775807")

    summary = ledger.summary()
    assert summary["blocked_on_overrun"] is True
    assert summary["actual_usd"] == "9223372036855.775807"
    assert summary["over_budget_usd"] == "9223372036850.775807"


def test_conflicting_reservation_identity_and_limit_are_rejected(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = BudgetLedger(path, limit_usd="2")
    ledger.reserve("id-1", "0.5", {"model": "a"})

    with pytest.raises(IdempotencyConflict):
        ledger.reserve("id-1", "0.6", {"model": "a"})
    with pytest.raises(IdempotencyConflict):
        ledger.reserve("id-1", "0.5", {"model": "b"})
    with pytest.raises(ValueError):
        BudgetLedger(path, limit_usd="3")


@pytest.mark.parametrize("amount", ["-0.01", "NaN", "Infinity"])
def test_invalid_money_amounts_are_rejected(tmp_path, amount):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_usd="2")
    with pytest.raises(ValueError):
        ledger.reserve("bad", amount)


def test_cost_estimate_uses_byte_bound_and_full_confirmed_output_cap():
    estimate = estimate_max_cost(
        prompt_bytes=100,
        max_tokens=1000,
        pricing=Pricing("0.000001", "0.000001", request_fee_usd="0.002"),
        provider_max_tokens_confirmed=True,
    )

    assert estimate.input_token_upper_bound == 4196
    assert estimate.output_token_cap == 1000
    assert estimate.maximum_usd == Decimal("0.007196")
    assert "4096" in estimate.assumptions
    assert "reasoning" in estimate.assumptions.lower()


def test_cost_estimate_requires_a_confirmed_provider_output_cap():
    with pytest.raises(ValueError, match="confirmed"):
        estimate_max_cost(100, 100, Pricing("0.000001", "0.000001", "0"),
                          provider_max_tokens_confirmed=False)


def test_unknown_request_fee_is_not_treated_as_zero():
    with pytest.raises(ValueError, match="request_fee_usd"):
        Pricing("0.000001", "0.000001", None)


def test_cost_estimate_rounds_up_to_micro_usd():
    estimate = estimate_max_cost(
        0, 1, Pricing("0.0000001", "0", request_fee_usd="0"),
        provider_max_tokens_confirmed=True,
    )
    assert estimate.maximum_usd == Decimal("0.000410")
