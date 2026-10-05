"""Multi-factor momentum/trend model: scores, targets and live recommendations.

Everything here is computed from daily bars only and uses data available at the
close of day t. The backtester (vendor/finance) executes the resulting targets at
the next open, so there is no look-ahead.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import BENCHMARK, Panel


@dataclass
class Params:
    top_n: int = 10             # target number of positions
    max_weight: float = 0.15    # cap per position (fraction of equity)
    rebalance_days: int = 5     # trading days between rebalances (backtest)
    hold_rank_mult: float = 2.0 # keep a holding while rank <= top_n * mult (reduces churn)
    stop_loss: float = 0.12     # sell if price is this far below cost basis
    atr_stop_mult: float = 2.5  # suggested protective stop = price - mult * ATR
    regime_filter: bool = True  # halve exposure when benchmark < its 200d average
    min_history: int = 260
    min_price: float = 5.0          # liquidity guards: never pick penny / illiquid stocks
    min_dollar_vol: float = 5e6     # median 60d daily $ volume


@dataclass
class Factors:
    score: pd.DataFrame
    eligible: pd.DataFrame
    vol: pd.DataFrame
    sma50: pd.DataFrame
    sma200: pd.DataFrame
    rsi14: pd.DataFrame
    macd_hist: pd.DataFrame
    atr14: pd.DataFrame
    mom12_1: pd.DataFrame
    mom3: pd.DataFrame
    regime_on: pd.Series = field(default=None)
    liquid: pd.DataFrame = field(default=None)


def _wilder(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder smoothing across all columns at once (same recursion as finance.indicators.smma)."""
    return df.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi_panel(c: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), window), _wilder(-d.clip(upper=0), window)
    tot = up + dn
    return (100 * up / tot.where(tot != 0)).mask(tot == 0, 50)


def atr_panel(h, l, c, window: int = 14) -> pd.DataFrame:
    pc = c.shift()
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return _wilder(tr, window)


def macd_hist_panel(c: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema = lambda x, n: x.ewm(span=n, adjust=False, min_periods=n).mean()  # noqa: E731
    line = ema(c, fast) - ema(c, slow)
    return line - ema(line, signal)


def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in [-0.5, 0.5] per date."""
    return df.rank(axis=1, pct=True) - 0.5


def compute_factors(panel: Panel, p: Params | None = None, keep=()) -> Factors:
    p = p or Params()
    c, h, l = panel.close, panel.high, panel.low
    bench = c[BENCHMARK]
    stocks = [t for t in c.columns if t != BENCHMARK]
    c, h, l = c[stocks], h[stocks], l[stocks]
    # only compute indicators for names that were liquid recently (thousands of columns otherwise)
    dollar_vol = (c * panel.volume[stocks]).rolling(60, min_periods=30).median()
    liquid = (c >= p.min_price) & (dollar_vol >= p.min_dollar_vol)
    ever = liquid.tail(250).any() & (c.count() >= p.min_history)
    stocks = [t for t in stocks if ever[t] or t in keep]
    c, h, l, liquid = c[stocks], h[stocks], l[stocks], liquid[stocks]

    mom12_1 = c.shift(21) / c.shift(252) - 1
    mom6 = c.shift(5) / c.shift(126) - 1
    mom3 = c / c.shift(63) - 1
    rel = mom12_1.sub(bench.shift(21) / bench.shift(252) - 1, axis=0)
    vol = np.log(c).diff().rolling(60).std() * np.sqrt(252)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    rsi14 = rsi_panel(c)
    hist = macd_hist_panel(c) / c  # normalised by price
    atr14 = atr_panel(h, l, c)

    score = (
        0.35 * _xs_rank(mom12_1)
        + 0.20 * _xs_rank(mom6)
        + 0.15 * _xs_rank(mom3)
        + 0.15 * _xs_rank(rel)
        - 0.10 * _xs_rank(vol)          # prefer calmer risers
        + 0.05 * _xs_rank(hist)
    )
    score = score.where(rsi14 < 85, score - 0.15)  # fade blow-off spikes
    eligible = (c > s200) & (s50 > s200) & (mom12_1 > 0) & score.notna() & liquid
    regime = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
    return Factors(score, eligible, vol, s50, s200, rsi14, hist, atr14, mom12_1, mom3, regime, liquid)


def target_weights(f: Factors, panel: Panel, p: Params, start: int = 0) -> pd.DataFrame:
    """Day-by-day target weights (signed fractions, sum <= 1) for the backtester."""
    idx = f.score.index
    cols = f.score.columns
    out = np.zeros((len(idx), len(cols)))
    held: set[int] = set()
    cur = np.zeros(len(cols))
    score, elig = f.score.to_numpy(), f.eligible.to_numpy()
    vol = f.vol.to_numpy()
    regime = f.regime_on.to_numpy()
    for i in range(len(idx)):
        if i >= start and (i - start) % p.rebalance_days == 0 and np.isfinite(score[i]).any():
            s = np.where(np.isfinite(score[i]), score[i], -np.inf)
            order = np.argsort(-s)
            rank = np.empty(len(s), dtype=int)
            rank[order] = np.arange(len(s))
            keep = {j for j in held if elig[i, j] and rank[j] < p.top_n * p.hold_rank_mult}
            picks = list(keep)
            for j in order:
                if len(picks) >= p.top_n:
                    break
                if elig[i, j] and j not in keep:
                    picks.append(j)
            held = set(picks)
            cur = np.zeros(len(cols))
            if picks:
                inv = 1 / np.clip(vol[i, picks], 0.05, None)
                w = inv / inv.sum()
                # equity fraction invested: full when top_n filled, scaled by regime
                exposure = (len(picks) / p.top_n) * (1.0 if (regime[i] or not p.regime_filter) else 0.5)
                w = np.minimum(w * exposure, p.max_weight)
                # keep existing sizes unless they drifted a lot: avoids paying costs to
                # nudge every position each week
                new = np.zeros(len(cols))
                new[picks] = w
                for j in picks:
                    if cur[j] > 0 and abs(new[j] - cur[j]) < 0.03:
                        new[j] = cur[j]
                if new.sum() > 1:
                    new /= new.sum()
                cur = new
        out[i] = cur
    return pd.DataFrame(out, index=idx, columns=cols)


# ----------------------------------------------------------------- live picks

def recommend(panel: Panel, meta: dict, portfolio: dict, p: Params | None = None) -> dict:
    p = p or Params()
    f = compute_factors(panel, p, keep=tuple(x["ticker"] for x in portfolio.get("positions", [])))
    last = f.score.index[-1]
    close = panel.close.iloc[-1]
    score = f.score.loc[last].dropna().sort_values(ascending=False)
    rank = {t: i + 1 for i, t in enumerate(score.index)}
    n = len(score)
    regime_on = bool(f.regime_on.loc[last])

    positions = {x["ticker"]: x for x in portfolio.get("positions", [])}
    cash = float(portfolio.get("cash", 0))
    value = {t: x["shares"] * float(close[t]) for t, x in positions.items() if t in close.index}
    equity = cash + sum(value.values())
    if equity <= 0:
        equity = 10000.0  # nothing entered yet -> size picks against a $10k example account
        example = True
    else:
        example = False

    def row(t):
        return {
            "ticker": t,
            "name": meta.get(t, {}).get("name", t),
            "sector": meta.get(t, {}).get("sector", ""),
            "price": round(float(close[t]), 2),
            "score": round(float(score.get(t, np.nan)), 3) if t in score else None,
            "rank": rank.get(t),
            "mom_12_1": _pct(f.mom12_1.at[last, t]),
            "mom_3m": _pct(f.mom3.at[last, t]),
            "rsi": _r(f.rsi14.at[last, t], 1),
            "volatility": _pct(f.vol.at[last, t]),
            "above_200d": bool(close[t] > f.sma200.at[last, t]) if np.isfinite(f.sma200.at[last, t]) else None,
            "stop": round(float(close[t] - p.atr_stop_mult * f.atr14.at[last, t]), 2)
            if np.isfinite(f.atr14.at[last, t]) else None,
        }

    recs: list[dict] = []
    # 1) review what you already hold
    for t, pos in positions.items():
        if t not in close.index or not np.isfinite(close[t]):
            recs.append({"ticker": t, "action": "REVIEW", "shares": 0, "name": t,
                         "reasons": ["No market data found for this ticker."]})
            continue
        r = row(t)
        cost = float(pos.get("cost") or 0)
        pnl = (r["price"] / cost - 1) if cost > 0 else None
        w = value[t] / equity
        reasons, action = [], "HOLD"
        if pnl is not None and pnl <= -p.stop_loss:
            action, reasons = "SELL", [f"Down {pnl:.1%} vs your cost basis (stop-loss {p.stop_loss:.0%})."]
        elif t not in f.score.columns or not bool(f.liquid.at[last, t]):
            action = "REVIEW"
            reasons = [f"Below the model's liquidity floor (price ≥ ${p.min_price:g}, "
                       f"${p.min_dollar_vol / 1e6:g}M/day traded) or too little history; not scored."]
        elif not f.eligible.at[last, t]:
            action = "SELL"
            reasons = ["Trend broke: fails price > 200d avg, 50d > 200d, or 12-1m momentum > 0."]
        elif rank.get(t, n) > p.top_n * p.hold_rank_mult:
            action = "SELL"
            reasons = [f"Momentum rank fell to #{rank.get(t)} of {n}; better candidates exist."]
        elif w > p.max_weight * 1.3:
            action = "TRIM"
            reasons = [f"Position is {w:.0%} of portfolio (cap {p.max_weight:.0%})."]
        else:
            reasons = [f"Still ranks #{rank.get(t)} of {n} with an intact uptrend."]
        sh = pos["shares"]
        if action == "TRIM":
            sh = int(max(1, (value[t] - p.max_weight * equity) // r["price"]))
        r.update(action=action, shares=sh if action != "HOLD" else 0, held=pos["shares"],
                 weight=round(w, 4), pnl=_r(pnl * 100, 1) if pnl is not None else None,
                 reasons=reasons)
        recs.append(r)

    # 2) new buys from the top of the ranking
    selling = {r["ticker"]: r for r in recs if r["action"] == "SELL"}
    kept = [t for t in positions if t not in selling]
    slots = max(0, p.top_n - len(kept))
    freed = cash + sum(value[t] for t in selling if t in value)
    budget = freed - (0 if regime_on or not p.regime_filter else 0.5 * equity)
    cand = [t for t in score.index if f.eligible.at[last, t] and t not in positions]
    picks = cand[:slots]
    if picks:
        inv = np.array([1 / max(float(f.vol.at[last, t]), 0.05) for t in picks])
        weights = np.minimum(inv / inv.sum() * (len(picks) / p.top_n), p.max_weight)
        for t, w in zip(picks, weights):
            r = row(t)
            dollars = min(w * equity, max(budget, 0))
            sh = int(dollars // r["price"])
            if sh < 1:
                continue
            budget -= sh * r["price"]
            r.update(action="BUY", shares=sh, dollars=round(sh * r["price"], 2),
                     weight=round(sh * r["price"] / equity, 4), held=0,
                     reasons=[f"Ranks #{rank[t]} of {n}: 12-1m momentum {r['mom_12_1']}%, "
                              f"3m {r['mom_3m']}%, uptrend intact (price > 200d avg).",
                              f"Suggested stop ≈ ${r['stop']} (price − {p.atr_stop_mult}×ATR)."])
            recs.append(r)
    order = {"SELL": 0, "TRIM": 1, "BUY": 2, "REVIEW": 3, "HOLD": 4}
    recs.sort(key=lambda r: (order[r["action"]], r.get("rank") or 999))
    leaderboard = [row(t) for t in score.index[:25]]
    return {
        "asof": panel.asof, "source": panel.source, "synthetic": panel.synthetic,
        "market_regime": "risk-on (S&P 500 ETF above 200d average)" if regime_on
        else "risk-off (S&P 500 ETF below 200d average) — model keeps ~50% cash",
        "universe_size": n, "equity": round(equity, 2), "example_account": example,
        "recommendations": recs, "leaderboard": leaderboard,
    }


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _pct(x):
    return _r(x * 100, 1) if x is not None and np.isfinite(x) else None
