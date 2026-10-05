"""Paper-money backtest: replay the model over history with a virtual cash account."""

import numpy as np
import pandas as pd

from finance.backtesting import backtest
from finance.analytics import performance

from .data import BENCHMARK, Panel
from .strategy import Params, compute_factors, target_weights


def run_backtest(panel: Panel, params: Params, years: float = 3.0, initial_cash: float = 10000,
                 commission: float = 0.001, slippage: float = 0.0005) -> dict:
    stocks = [t for t in panel.close.columns if t != BENCHMARK]
    f = compute_factors(panel, params)
    idx = panel.close.index
    start_date = idx[-1] - pd.Timedelta(days=int(365.25 * years))
    start = int(idx.searchsorted(start_date))
    start = max(start, params.min_history)
    if len(idx) - start < 60:
        raise ValueError("Not enough price history for this window; lower the years.")

    # engine needs gap-free prices: keep tickers that traded over the whole test window
    window = slice(start, None)
    ok = [t for t in stocks if panel.close[t].iloc[window].notna().all()
          and panel.open[t].iloc[window].notna().all()]
    if len(ok) < params.top_n:
        raise ValueError(f"Only {len(ok)} tickers have full history in this window.")
    f.score, f.eligible = f.score[ok], f.eligible[ok]
    f.vol = f.vol[ok]
    weights = target_weights(f, panel.subset(ok), params, start=start).iloc[window]
    o, c = panel.open[ok].iloc[window], panel.close[ok].iloc[window]
    res = backtest(o, c, weights, initial_cash=initial_cash, commission=commission,
                   slippage=slippage, liquidate=False)

    bench_close, bench_open = panel.close[BENCHMARK].iloc[window], panel.open[BENCHMARK].iloc[window]
    # buy & hold the benchmark with the same costs: buy at first open, mark at close
    shares = initial_cash * (1 - commission) / (bench_open.iloc[0] * (1 + slippage))
    bench_equity = shares * bench_close
    bench_ret = bench_equity.pct_change()
    bench_ret.iloc[0] = bench_equity.iloc[0] / initial_cash - 1
    bm = performance(bench_ret)

    m = res.metrics
    trades = res.trades.copy()
    holdings = res.holdings.iloc[-1]
    last_close = c.iloc[-1]
    open_pos = [{"ticker": t, "shares": round(float(s), 3), "value": round(float(s * last_close[t]), 2)}
                for t, s in holdings.items() if abs(s) > 1e-9]
    open_pos.sort(key=lambda x: -x["value"])
    gross = (res.holdings * c).abs().sum(axis=1) / res.equity
    n = len(res.equity)
    step = max(1, n // 400)  # downsample for the chart
    sel = list(range(0, n, step)) + ([n - 1] if (n - 1) % step else [])
    dates = [str(res.equity.index[i].date()) for i in sel]
    dd = res.equity / res.equity.cummax() - 1
    return {
        "source": panel.source, "synthetic": panel.synthetic,
        "start": str(res.equity.index[0].date()), "end": str(res.equity.index[-1].date()),
        "initial_cash": initial_cash, "final_equity": round(float(res.equity.iloc[-1]), 2),
        "universe_tested": len(ok),
        "metrics": {
            "total_return": _f(m["total_return"]), "cagr": _f(m["cagr"]),
            "volatility": _f(m["volatility"]), "sharpe": _f(m["sharpe"]),
            "sortino": _f(m["sortino"]), "max_drawdown": _f(m["max_drawdown"]),
            "commissions": round(float(m["commissions"]), 2), "fills": int(m["fill_count"]),
            "avg_exposure": _f(gross.mean()),
        },
        "benchmark": {"name": BENCHMARK, "total_return": _f(bm["total_return"]),
                      "cagr": _f(bm["cagr"]), "sharpe": _f(bm["sharpe"]),
                      "max_drawdown": _f(bm["max_drawdown"]),
                      "final_equity": round(float(bench_equity.iloc[-1]), 2)},
        "curve": {"dates": dates,
                  "equity": [round(float(res.equity.iloc[i]), 2) for i in sel],
                  "benchmark": [round(float(bench_equity.iloc[i]), 2) for i in sel],
                  "drawdown": [round(float(dd.iloc[i]) * 100, 2) for i in sel]},
        "trades": [{"date": str(r.date.date()), "ticker": r.asset,
                    "side": "BUY" if r.quantity > 0 else "SELL",
                    "shares": round(abs(float(r.quantity)), 3), "price": round(float(r.price), 2)}
                   for r in trades.itertuples()][-300:],
        "open_positions": open_pos,
        "cash": round(float(res.cash.iloc[-1]), 2),
        "notes": ["Universe is today's US mega-caps (survivorship bias flatters results).",
                  "Signals use data through each close; fills happen at the next open with "
                  f"{commission:.2%} commission and {slippage:.2%} slippage.",
                  "Past performance does not predict future returns."],
    }


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)
