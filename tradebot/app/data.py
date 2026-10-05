"""Market data: real daily prices from Yahoo Finance (cached on disk).

`TRADEBOT_DEMO=1` switches to clearly-labelled SYNTHETIC prices so the UI can be
exercised offline. Real mode never falls back to fake data.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.environ.get("TRADEBOT_CACHE", Path(__file__).resolve().parents[1] / ".cache"))
CACHE_TTL = 6 * 3600
BENCHMARK = "SPY"
_lock = threading.Lock()


class DataError(RuntimeError):
    pass


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    source: str
    synthetic: bool

    @property
    def asof(self) -> str:
        return str(self.close.index[-1].date())

    def subset(self, tickers) -> "Panel":
        cols = [t for t in tickers if t in self.close.columns]
        return Panel(*(getattr(self, f)[cols] for f in ("open", "high", "low", "close", "volume")),
                     self.source, self.synthetic)


def demo_mode() -> bool:
    return os.environ.get("TRADEBOT_DEMO") == "1"


def _synthetic(tickers: list[str], years: float) -> Panel:
    n = int(252 * years)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rng = np.random.default_rng(42)
    market = rng.normal(0.0004, 0.01, n)
    frames = {f: {} for f in ("open", "high", "low", "close", "volume")}
    for t in tickers:
        r = 0.9 * market + rng.normal(rng.normal(0.0003, 0.0004), rng.uniform(0.008, 0.02), n)
        close = 100 * np.exp(np.cumsum(r))
        opn = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
        hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        for f, v in zip(frames, (opn, hi, lo, close, rng.integers(1e6, 5e7, n).astype(float))):
            frames[f][t] = v
    return Panel(*(pd.DataFrame(frames[f], index=idx) for f in frames), "SYNTHETIC DEMO DATA", True)


def _download(tickers: list[str], years: float) -> Panel:
    import yfinance as yf

    start = (pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data (network blocked or rate limited?)")
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers[:1]])
    fields = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        df = raw[f].copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        fields[f.lower()] = df.dropna(axis=1, how="all")
    # keep only symbols with usable OHLC; drop rows where the benchmark didn't trade
    cols = [c for c in fields["close"].columns
            if all(c in fields[f].columns for f in fields)]
    panel = [fields[f][cols].sort_index() for f in ("open", "high", "low", "close", "volume")]
    return Panel(*panel, "Yahoo Finance (split/dividend adjusted)", False)


def get_panel(tickers: list[str], years: float = 5.0, refresh: bool = False) -> Panel:
    """Daily OHLCV for `tickers` + benchmark. Cached for CACHE_TTL seconds."""
    wanted = sorted(set(tickers) | {BENCHMARK})
    if demo_mode():
        return _synthetic(wanted, years)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(",".join(wanted).encode()).hexdigest()[:10]
    cache = CACHE_DIR / f"prices-{key}-{years:g}y.pkl"
    with _lock:
        if not refresh and cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL:
            try:
                p: Panel = pd.read_pickle(cache)
                return p
            except Exception:
                pass  # corrupt cache -> re-download
        p = _download(wanted, years)
        if BENCHMARK not in p.close.columns:
            raise DataError(f"benchmark {BENCHMARK} missing from download")
        pd.to_pickle(p, cache)
        return p
