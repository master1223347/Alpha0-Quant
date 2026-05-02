"""Execution-cost helpers for intraday backtests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, time
from typing import Any


@dataclass(slots=True)
class ExecutionCostBreakdown:
    spread_bps: float
    impact_bps: float
    total_cost_bps: float
    max_pov: float
    requested_pov: float
    filled_fraction: float
    bar_type: str
    penalty_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None
    if hasattr(value, "timestamp"):
        try:
            return datetime.fromtimestamp(float(value.timestamp()))
        except Exception:
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=None)
        except Exception:
            return None
    return None


def _bar_type_from_timestamp(timestamp: Any, *, bar_index: int, total_bars: int) -> str:
    parsed = _coerce_datetime(timestamp)
    if parsed is not None:
        current_time = parsed.time()
        if current_time <= time(9, 30):
            return "open_auction"
        if current_time >= time(15, 55):
            return "close_auction"
        return "regular"
    if bar_index <= 0:
        return "open_auction"
    if bar_index >= total_bars - 1:
        return "close_auction"
    return "regular"


def enforce_participation_caps(
    *,
    requested_pov: float,
    max_pov: float,
    reject_excess: bool = False,
) -> float:
    """Return filled fraction after max participation cap."""
    requested = max(0.0, float(requested_pov))
    cap = max(0.0, float(max_pov))
    if requested <= 1e-12:
        return 0.0
    if requested <= cap:
        return 1.0
    if reject_excess:
        return 0.0
    return cap / requested if requested > 0 else 0.0


def apply_open_close_auction_rules(
    *,
    timestamp: Any,
    bar_index: int,
    total_bars: int,
    regular_max_pov: float,
    open_max_pov: float,
    close_max_pov: float,
    open_penalty_bars: int,
    close_penalty_bars: int,
    use_open_auction: bool,
    use_close_auction: bool,
) -> tuple[str, float, float]:
    """Return `(bar_type, max_pov, penalty_multiplier)`."""
    bar_type = _bar_type_from_timestamp(timestamp, bar_index=bar_index, total_bars=total_bars)
    if bar_type == "open_auction" and use_open_auction:
        return bar_type, float(open_max_pov), 1.0
    if bar_type == "close_auction" and use_close_auction:
        return bar_type, float(close_max_pov), 1.0

    penalty = 1.0
    if bar_index < max(0, int(open_penalty_bars)):
        penalty = max(penalty, 1.5)
    if bar_index >= max(0, total_bars - int(close_penalty_bars)):
        penalty = max(penalty, 1.35)
    return "regular", float(regular_max_pov), penalty


class LiquidityBucketCostModel:
    """Heuristic spread + impact model keyed by volatility/liquidity bucket."""

    def __init__(
        self,
        *,
        base_spread_bps: float = 6.0,
        base_impact_bps: float = 25.0,
        median_sigma_5m: float = 0.001,
    ) -> None:
        self.base_spread_bps = float(base_spread_bps)
        self.base_impact_bps = float(base_impact_bps)
        self.median_sigma_5m = max(float(median_sigma_5m), 1e-8)

    def estimate(self, *, sigma_5m: float, requested_pov: float, penalty_multiplier: float) -> tuple[float, float]:
        sigma = max(float(sigma_5m), 1e-8)
        sigma_multiplier = math.sqrt(sigma / self.median_sigma_5m)
        spread_bps = self.base_spread_bps * sigma_multiplier * float(penalty_multiplier)
        impact_bps = (self.base_impact_bps * max(0.0, float(requested_pov))) + (10000.0 * sigma * 0.05)
        impact_bps *= float(penalty_multiplier)
        return spread_bps, impact_bps


class AuctionExecutionModel:
    """Combined auction/regular-bar execution model."""

    def __init__(
        self,
        *,
        use_open_auction: bool = True,
        use_close_auction: bool = True,
        regular_max_pov: float = 0.05,
        open_max_pov: float = 0.03,
        close_max_pov: float = 0.05,
        open_penalty_bars: int = 3,
        close_penalty_bars: int = 3,
        base_spread_bps: float = 6.0,
        base_impact_bps: float = 25.0,
        reject_excess_pov: bool = False,
    ) -> None:
        self.use_open_auction = bool(use_open_auction)
        self.use_close_auction = bool(use_close_auction)
        self.regular_max_pov = float(regular_max_pov)
        self.open_max_pov = float(open_max_pov)
        self.close_max_pov = float(close_max_pov)
        self.open_penalty_bars = int(open_penalty_bars)
        self.close_penalty_bars = int(close_penalty_bars)
        self.reject_excess_pov = bool(reject_excess_pov)
        self.cost_model = LiquidityBucketCostModel(
            base_spread_bps=base_spread_bps,
            base_impact_bps=base_impact_bps,
        )

    def estimate_cost(
        self,
        *,
        timestamp: Any,
        bar_index: int,
        total_bars: int,
        requested_pov: float,
        sigma_5m: float,
    ) -> ExecutionCostBreakdown:
        bar_type, max_pov, penalty_multiplier = apply_open_close_auction_rules(
            timestamp=timestamp,
            bar_index=bar_index,
            total_bars=total_bars,
            regular_max_pov=self.regular_max_pov,
            open_max_pov=self.open_max_pov,
            close_max_pov=self.close_max_pov,
            open_penalty_bars=self.open_penalty_bars,
            close_penalty_bars=self.close_penalty_bars,
            use_open_auction=self.use_open_auction,
            use_close_auction=self.use_close_auction,
        )
        filled_fraction = enforce_participation_caps(
            requested_pov=requested_pov,
            max_pov=max_pov,
            reject_excess=self.reject_excess_pov,
        )
        spread_bps, impact_bps = self.cost_model.estimate(
            sigma_5m=sigma_5m,
            requested_pov=min(float(requested_pov), max_pov),
            penalty_multiplier=penalty_multiplier,
        )
        return ExecutionCostBreakdown(
            spread_bps=float(spread_bps),
            impact_bps=float(impact_bps),
            total_cost_bps=float(spread_bps + impact_bps),
            max_pov=float(max_pov),
            requested_pov=float(requested_pov),
            filled_fraction=float(filled_fraction),
            bar_type=bar_type,
            penalty_multiplier=float(penalty_multiplier),
        )


def simulate_execution_costs(
    *,
    timestamp: Any,
    bar_index: int,
    total_bars: int,
    requested_pov: float,
    sigma_5m: float,
    use_open_auction: bool = True,
    use_close_auction: bool = True,
    regular_max_pov: float = 0.05,
    open_max_pov: float = 0.03,
    close_max_pov: float = 0.05,
    open_penalty_bars: int = 3,
    close_penalty_bars: int = 3,
    base_spread_bps: float = 6.0,
    base_impact_bps: float = 25.0,
    reject_excess_pov: bool = False,
) -> ExecutionCostBreakdown:
    model = AuctionExecutionModel(
        use_open_auction=use_open_auction,
        use_close_auction=use_close_auction,
        regular_max_pov=regular_max_pov,
        open_max_pov=open_max_pov,
        close_max_pov=close_max_pov,
        open_penalty_bars=open_penalty_bars,
        close_penalty_bars=close_penalty_bars,
        base_spread_bps=base_spread_bps,
        base_impact_bps=base_impact_bps,
        reject_excess_pov=reject_excess_pov,
    )
    return model.estimate_cost(
        timestamp=timestamp,
        bar_index=bar_index,
        total_bars=total_bars,
        requested_pov=requested_pov,
        sigma_5m=sigma_5m,
    )
