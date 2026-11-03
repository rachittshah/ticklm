"""Transactional paid-call budget reservations in conservative micro-USD."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path

_USD_SCALE = 1_000_000
_MAX_SQLITE_INT = (1 << 63) - 1
_INPUT_TOKEN_OVERHEAD = 4096


class BudgetLedgerError(RuntimeError):
    """Base class for safe budget-ledger refusals."""


class BudgetExceeded(BudgetLedgerError):
    """A reservation would exceed the configured spend limit."""


class BudgetBlocked(BudgetLedgerError):
    """A provider exceeded a reservation, so further paid calls are blocked."""


class IdempotencyConflict(BudgetLedgerError):
    """An explicit request ID was reused with different reservation data."""


class UnknownReservation(BudgetLedgerError):
    """Settlement was requested for an ID that has never been reserved."""


@dataclass(frozen=True)
class Reservation:
    request_id: str
    max_usd: Decimal
    metadata: dict
    state: str
    created: bool

    @property
    def can_execute(self) -> bool:
        """Only the call that atomically created the reservation may execute."""
        return self.created


@dataclass(frozen=True)
class Settlement:
    request_id: str
    charged_usd: Decimal
    actual_usd: Decimal | None
    unknown: bool
    exceeded_reservation: bool
    overrun_usd: Decimal
    blocked: bool


@dataclass(frozen=True)
class Pricing:
    """Text prices in USD per token plus an explicit fixed request fee."""

    input_usd_per_token: Decimal
    output_usd_per_token: Decimal
    request_fee_usd: Decimal

    def __init__(self, input_usd_per_token, output_usd_per_token, request_fee_usd):
        object.__setattr__(self, "input_usd_per_token", _decimal_price(
            input_usd_per_token, "input_usd_per_token"))
        object.__setattr__(self, "output_usd_per_token", _decimal_price(
            output_usd_per_token, "output_usd_per_token"))
        object.__setattr__(self, "request_fee_usd", _decimal_price(
            request_fee_usd, "request_fee_usd"))


@dataclass(frozen=True)
class CostEstimate:
    input_token_upper_bound: int
    output_token_cap: int
    input_cost_usd: Decimal
    output_cost_usd: Decimal
    request_fee_usd: Decimal
    maximum_usd: Decimal
    assumptions: str


def _decimal_amount(value, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite USD amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite USD amount") from exc
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        qualifier = "positive and finite" if positive else "nonnegative and finite"
        raise ValueError(f"{name} must be {qualifier}")
    return amount


def _decimal_price(value, name: str) -> Decimal:
    return _decimal_amount(value, name)


def _to_micros(value, name: str, *, positive: bool = False) -> int:
    amount = _decimal_amount(value, name, positive=positive)
    micros = int((amount * _USD_SCALE).to_integral_value(rounding=ROUND_CEILING))
    if micros > _MAX_SQLITE_INT:
        raise ValueError(f"{name} exceeds the ledger's supported range")
    return micros


def _from_micros(value: int) -> Decimal:
    return Decimal(value) / _USD_SCALE


def _money_text(value: int) -> str:
    return f"{_from_micros(value):.6f}"


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 256:
        raise ValueError("request_id must be a nonempty explicit identifier of at most 256 chars")


def _metadata_json(metadata: Mapping | None) -> str:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a JSON-compatible mapping")
    try:
        return json.dumps(dict(metadata), sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be a JSON-compatible mapping") from exc


class BudgetLedger:
    """Reserve and settle paid-call envelopes without concurrent overspend.

    Amounts round upward to one micro-USD. An unknown settlement retains its
    reservation until a later call supplies the actual charge. Metadata must
    contain safe run identifiers only; callers must never store credentials.
    """

    def __init__(self, path: str | Path, limit_usd=Decimal(25)):
        raw_path = str(path)
        if raw_path == ":memory:":
            raise ValueError("BudgetLedger requires a durable SQLite path")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._limit_micros = _to_micros(limit_usd, "limit_usd", positive=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS budget_meta ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                "limit_micros INTEGER NOT NULL, blocked_on_overrun INTEGER NOT NULL DEFAULT 0, "
                "created_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS budget_requests ("
                "request_id TEXT PRIMARY KEY, reserve_micros INTEGER NOT NULL CHECK(reserve_micros > 0), "
                "actual_micros INTEGER, state TEXT NOT NULL CHECK(state IN ('reserved', 'unknown', 'settled')), "
                "metadata_json TEXT NOT NULL, reservation_overrun_micros INTEGER NOT NULL DEFAULT 0, "
                "created_at TEXT NOT NULL, settled_at TEXT)"
            )
            existing = connection.execute(
                "SELECT limit_micros FROM budget_meta WHERE singleton = 1"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO budget_meta (singleton, limit_micros, created_at) VALUES (1, ?, ?)",
                    (self._limit_micros, _now()),
                )
            elif existing["limit_micros"] != self._limit_micros:
                raise ValueError("limit_usd does not match the existing ledger")

    def reserve(self, request_id: str, max_usd, metadata: Mapping | None = None) -> Reservation:
        """Create one spend reservation; duplicate IDs are never executable again."""
        _validate_request_id(request_id)
        reserve_micros = _to_micros(max_usd, "max_usd", positive=True)
        metadata_text = _metadata_json(metadata)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT request_id, reserve_micros, state, metadata_json "
                "FROM budget_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (existing["reserve_micros"] != reserve_micros
                        or existing["metadata_json"] != metadata_text):
                    raise IdempotencyConflict("request_id was already used with different reservation data")
                return Reservation(request_id, _from_micros(reserve_micros),
                                   json.loads(metadata_text), existing["state"], False)

            meta = connection.execute(
                "SELECT limit_micros, blocked_on_overrun FROM budget_meta WHERE singleton = 1"
            ).fetchone()
            if meta["blocked_on_overrun"]:
                raise BudgetBlocked("ledger is blocked after an actual-cost reservation overrun")
            totals = self._totals(connection)
            committed = totals["actual"] + totals["reserved"] + totals["unknown"]
            remaining = max(0, meta["limit_micros"] - committed)
            if reserve_micros > remaining:
                raise BudgetExceeded(
                    f"reservation requires {_money_text(reserve_micros)} USD; "
                    f"only {_money_text(remaining)} USD remains"
                )
            connection.execute(
                "INSERT INTO budget_requests "
                "(request_id, reserve_micros, actual_micros, state, metadata_json, created_at) "
                "VALUES (?, ?, NULL, 'reserved', ?, ?)",
                (request_id, reserve_micros, metadata_text, _now()),
            )
            return Reservation(request_id, _from_micros(reserve_micros),
                               json.loads(metadata_text), "reserved", True)

    def settle(self, request_id: str, actual_usd=None) -> Settlement:
        """Settle a request, retaining its full reservation when cost is unknown."""
        _validate_request_id(request_id)
        actual_micros = None if actual_usd is None else _to_micros(actual_usd, "actual_usd")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT reserve_micros, actual_micros, state, reservation_overrun_micros "
                "FROM budget_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise UnknownReservation("request_id has no recorded reservation")

            reserve_micros = row["reserve_micros"]
            if row["state"] == "settled":
                if actual_micros is not None and actual_micros != row["actual_micros"]:
                    raise IdempotencyConflict("request_id was already settled with a different actual cost")
                blocked = self._blocked(connection)
                actual = row["actual_micros"]
                overrun = row["reservation_overrun_micros"]
                return Settlement(request_id, _from_micros(actual), _from_micros(actual), False,
                                  overrun > 0, _from_micros(overrun), blocked)

            if actual_micros is None:
                if row["state"] == "reserved":
                    connection.execute(
                        "UPDATE budget_requests SET state = 'unknown', settled_at = ? WHERE request_id = ?",
                        (_now(), request_id),
                    )
                blocked = self._blocked(connection)
                return Settlement(request_id, _from_micros(reserve_micros), None, True,
                                  False, Decimal(0), blocked)

            overrun = max(0, actual_micros - reserve_micros)
            if overrun:
                connection.execute(
                    "UPDATE budget_meta SET blocked_on_overrun = 1 WHERE singleton = 1"
                )
            connection.execute(
                "UPDATE budget_requests SET actual_micros = ?, state = 'settled', "
                "reservation_overrun_micros = ?, settled_at = ? WHERE request_id = ?",
                (actual_micros, overrun, _now(), request_id),
            )
            return Settlement(request_id, _from_micros(actual_micros),
                              _from_micros(actual_micros), False, overrun > 0,
                              _from_micros(overrun), self._blocked(connection))

    def _blocked(self, connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            "SELECT blocked_on_overrun FROM budget_meta WHERE singleton = 1"
        ).fetchone()
        return bool(row["blocked_on_overrun"])

    @staticmethod
    def _totals(connection: sqlite3.Connection) -> dict[str, int]:
        """Sum with Python integers so a reported provider overrun cannot overflow SQLite."""
        totals = {"actual": 0, "reserved": 0, "unknown": 0,
                  "reservation_overrun": 0, "reserved_count": 0,
                  "unknown_count": 0, "settled_count": 0}
        rows = connection.execute(
            "SELECT reserve_micros, actual_micros, state, reservation_overrun_micros "
            "FROM budget_requests"
        )
        for row in rows:
            state = row["state"]
            if state == "reserved":
                totals["reserved"] += row["reserve_micros"]
                totals["reserved_count"] += 1
            elif state == "unknown":
                totals["unknown"] += row["reserve_micros"]
                totals["unknown_count"] += 1
            else:
                totals["actual"] += row["actual_micros"]
                totals["settled_count"] += 1
            totals["reservation_overrun"] += row["reservation_overrun_micros"]
        return totals

    def summary(self) -> dict:
        """Return a JSON-safe exact-spend summary; unknown cost stays reserved."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            try:
                meta = connection.execute(
                    "SELECT limit_micros, blocked_on_overrun FROM budget_meta WHERE singleton = 1"
                ).fetchone()
                totals = self._totals(connection)
                actual = totals["actual"]
                reserved = totals["reserved"]
                unknown = totals["unknown"]
                committed = actual + reserved + unknown
                limit = meta["limit_micros"]
                result = {
                    "limit_usd": _money_text(limit),
                    "actual_usd": _money_text(actual),
                    "reserved_usd": _money_text(reserved),
                    "unknown_usd": _money_text(unknown),
                    "remaining_usd": _money_text(max(0, limit - committed)),
                    "reservation_overrun_usd": _money_text(totals["reservation_overrun"]),
                    "over_budget_usd": _money_text(max(0, committed - limit)),
                    "blocked_on_overrun": bool(meta["blocked_on_overrun"]),
                    "request_counts": {
                        "reserved": totals["reserved_count"] or 0,
                        "unknown": totals["unknown_count"] or 0,
                        "settled": totals["settled_count"] or 0,
                    },
                }
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise


def estimate_max_cost(prompt_bytes: int, max_tokens: int, pricing: Pricing, *,
                      provider_max_tokens_confirmed: bool) -> CostEstimate:
    """Estimate a conservative text-call ceiling after provider cap semantics are confirmed.

    Input is bounded as one token per prompt byte plus 4096 tokens for chat and
    hidden request overhead. The output reserve covers the full advertised cap,
    including reasoning, only when the provider confirms that the cap covers all
    billable output tokens. This is still not a guarantee against provider billing
    errors or an adapter that fails to enforce the cap.
    """
    if isinstance(prompt_bytes, bool) or not isinstance(prompt_bytes, int) or prompt_bytes < 0:
        raise ValueError("prompt_bytes must be a nonnegative integer")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(provider_max_tokens_confirmed, bool):
        raise TypeError("provider_max_tokens_confirmed must be explicit")
    if not provider_max_tokens_confirmed:
        raise ValueError("a provider-confirmed billable output cap is required")
    if not isinstance(pricing, Pricing):
        raise TypeError("pricing must be a Pricing instance")
    input_bound = prompt_bytes + _INPUT_TOKEN_OVERHEAD
    input_cost = Decimal(input_bound) * pricing.input_usd_per_token
    output_cost = Decimal(max_tokens) * pricing.output_usd_per_token
    total = input_cost + output_cost + pricing.request_fee_usd
    total_micros = _to_micros(total, "estimated maximum cost")
    return CostEstimate(
        input_token_upper_bound=input_bound,
        output_token_cap=max_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        request_fee_usd=pricing.request_fee_usd,
        maximum_usd=_from_micros(total_micros),
        assumptions=("Input tokens are bounded by UTF-8 prompt bytes plus 4096 overhead tokens; "
                     "the provider-confirmed max_tokens cap covers all billable output, including "
                     "reasoning. The estimate is not protection against provider or adapter violations."),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
