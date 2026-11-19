import math

import pytest

from src.evaluation.backtest import run_backtest


def _interleaved_panel():
    # Time-major panel: at each timestamp AAA rises 1%, BBB falls 1%.
    probabilities, close, next_close, tickers, timestamps = [], [], [], [], []
    for step in range(4):
        for ticker, prob, move in (("AAA", 0.9, 1.01), ("BBB", 0.1, 0.99)):
            probabilities.append(prob)
            close.append(100.0)
            next_close.append(100.0 * move)
            tickers.append(ticker)
            timestamps.append(step)
    return probabilities, close, next_close, tickers, timestamps


def test_single_stream_lag_shifts_positions_by_one_bar():
    report = run_backtest(
        probabilities=[0.9, 0.9, 0.1],
        close=[100.0, 100.0, 100.0],
        next_close=[101.0, 99.0, 101.0],
        execution_lag_bars=1,
    )
    # Positions [0, +1, +1]: second bar loses 1%, third gains 1%.
    assert report.long_count == 2
    assert report.flat_count == 1
    assert report.trade_count == 1
    assert report.pnl == pytest.approx(0.0)


def test_lag_is_applied_within_each_ticker():
    probabilities, close, next_close, tickers, timestamps = _interleaved_panel()
    report = run_backtest(
        probabilities=probabilities,
        close=close,
        next_close=next_close,
        tickers=tickers,
        timestamps=timestamps,
        execution_lag_bars=1,
        cost_bps_per_trade=10.0,
    )
    # First bar of each ticker is flat; the next 3 bars hold AAA long and BBB short.
    assert report.long_count == 3
    assert report.short_count == 3
    assert report.flat_count == 2
    assert report.trade_count == 2
    assert report.gross_pnl == pytest.approx(6 * 0.01)
    assert report.transaction_cost_pnl == pytest.approx(-2 * 0.001)
    assert report.hit_rate == pytest.approx(1.0)


def test_without_tickers_interleaved_panel_mixes_streams():
    probabilities, close, next_close, _, _ = _interleaved_panel()
    report = run_backtest(probabilities=probabilities, close=close, next_close=next_close, execution_lag_bars=1)
    # Legacy single-stream behavior: each signal lands on the other ticker's row.
    assert report.gross_pnl < 0
    assert report.trade_count > 2


def test_mismatched_tickers_fall_back_to_single_stream():
    probabilities, close, next_close, _, _ = _interleaved_panel()
    legacy = run_backtest(probabilities=probabilities, close=close, next_close=next_close)
    mismatched = run_backtest(probabilities=probabilities, close=close, next_close=next_close, tickers=["AAA"])
    assert mismatched.pnl == pytest.approx(legacy.pnl)
    assert mismatched.trade_count == legacy.trade_count


def test_zero_lag_trades_on_signal_bar():
    report = run_backtest(
        probabilities=[0.9, 0.1],
        close=[100.0, 100.0],
        next_close=[101.0, 99.0],
        execution_lag_bars=0,
    )
    assert report.pnl == pytest.approx(0.02)
    assert report.trade_count == 2


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        run_backtest(probabilities=[0.5], close=[1.0, 2.0], next_close=[1.0, 2.0])
    with pytest.raises(ValueError):
        run_backtest(probabilities=[0.5], close=[1.0], next_close=[1.0], execution_lag_bars=-1)
    with pytest.raises(ValueError):
        run_backtest(probabilities=[], close=[], next_close=[])


def test_drawdown_and_sharpe_are_finite():
    report = run_backtest(
        probabilities=[0.9] * 6,
        close=[100.0] * 6,
        next_close=[101.0, 98.0, 101.0, 97.0, 102.0, 100.0],
        execution_lag_bars=0,
    )
    assert math.isfinite(report.sharpe)
    assert report.max_drawdown <= 0.0
    assert len(report.equity_curve) == 7
