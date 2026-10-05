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
PROGRESS = {"done": 0, "total": 0}
_mem: dict = {}


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


MIN_PRICE, MIN_DOLLAR_VOL, CHUNK = 5.0, 5e6, 400


def _fetch_chunk(tickers: list[str], start: str):
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or raw.empty:
        return None
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers[:1]])
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    return raw


def _liquid(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """Tradable now: recent price >= MIN_PRICE and median daily $ volume >= MIN_DOLLAR_VOL."""
    dv = (close * volume).tail(120).median()
    return (close.ffill().iloc[-1] >= MIN_PRICE) & (dv >= MIN_DOLLAR_VOL)


def _download(tickers: list[str], years: float, keep: frozenset = frozenset()) -> Panel:
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in ("Open", "High", "Low", "Close", "Volume")}
    PROGRESS.update(done=0, total=len(tickers))
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        PROGRESS["done"] = i
        try:
            raw = _fetch_chunk(chunk, start)
        except Exception:
            raw = None  # one bad batch must not sink the whole universe
        if raw is None:
            continue
        close = raw["Close"].dropna(axis=1, how="all")
        liquid = _liquid(close, raw["Volume"][close.columns])
        kept = close.columns[liquid | close.columns.isin(keep)]  # always keep user holdings
        for f in parts:  # drop illiquid names immediately to keep memory small
            parts[f].append(raw[f][kept].astype("float32"))
    if not parts["Close"]:
        raise DataError("Yahoo Finance returned no data (network blocked or rate limited?)")
    fields = {f.lower(): pd.concat(v, axis=1).sort_index() for f, v in parts.items()}
    cols = [c for c in fields["close"].columns if c in fields["open"].columns]
    # drop days when nothing traded (e.g. partial rows) and rows with all-NaN prices
    fields = {f: d[cols].dropna(how="all").astype(float) for f, d in fields.items()}
    idx = fields["close"].index
    fields = {f: d.reindex(idx) for f, d in fields.items()}
    return Panel(fields["open"], fields["high"], fields["low"], fields["close"], fields["volume"],
                 "Yahoo Finance (split/dividend adjusted)", False)


def get_panel(tickers: list[str], years: float = 5.0, refresh: bool = False,
              keep: tuple = ()) -> Panel:
    """Daily OHLCV for `tickers` + benchmark (+ `keep`: user holdings, exempt from the
    liquidity filter and fetched separately so the big universe cache is reused)."""
    wanted = sorted(set(tickers) | {BENCHMARK})
    return add_holdings(_universe_panel(wanted, years, refresh), keep, years)


def add_holdings(base: Panel, keep: tuple, years: float = 5.0) -> Panel:
    missing = [t for t in keep if t not in base.close.columns]
    if not missing or base.synthetic:
        return base
    try:
        extra = _download(missing, years, frozenset(missing))
    except DataError:
        return base  # holdings without data are flagged REVIEW downstream
    idx = base.close.index.union(extra.close.index)
    merged = [pd.concat([getattr(base, f).reindex(idx), getattr(extra, f).reindex(idx)], axis=1)
              for f in ("open", "high", "low", "close", "volume")]
    return Panel(*merged, base.source, base.synthetic)


def _universe_panel(wanted: list[str], years: float, refresh: bool) -> Panel:
    if demo_mode():
        return _synthetic(wanted, years)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(",".join(wanted).encode()).hexdigest()[:10]
    cache = CACHE_DIR / f"prices-{key}-{years:g}y.pkl"
    with _lock:
        hit = _mem.get(key)
        if not refresh and hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
        if not refresh and cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL:
            try:
                p = pd.read_pickle(cache)
                _mem[key] = (cache.stat().st_mtime, p)
                return p
            except Exception:
                pass  # corrupt cache -> re-download
        p = _download(wanted, years, frozenset({BENCHMARK}))
        if BENCHMARK not in p.close.columns:
            raise DataError(f"benchmark {BENCHMARK} missing from download")
        pd.to_pickle(p, cache)
        _mem[key] = (time.time(), p)
        return p
