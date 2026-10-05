"""FastAPI server: `python -m app` -> http://127.0.0.1:8000"""

import base64
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import portfolio
from .backtest import run_backtest
from .data import PROGRESS, DataError, add_holdings, get_panel
from .strategy import Params, recommend
from .universe import load_universe

STATIC = Path(__file__).resolve().parents[1] / "static"
app = FastAPI(title="Trade Suggestion Dashboard")
PASSWORD = os.environ.get("TRADEBOT_PASSWORD")  # set this on any public host
LIMIT = int(os.environ.get("TRADEBOT_UNIVERSE_LIMIT", "0")) or None  # smaller = less RAM
STATE = {"panel": None, "error": None, "universe": {}, "refresh": threading.Event()}


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if PASSWORD and request.url.path != "/healthz":
        sent = request.headers.get("authorization", "")
        try:
            given = base64.b64decode(sent.split(" ", 1)[1]).decode().split(":", 1)[1]
        except Exception:
            given = ""
        if not secrets.compare_digest(given.encode(), PASSWORD.encode()):
            return JSONResponse({"detail": "Password required"}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="Trade dashboard"'})
    return await call_next(request)


def _warm():
    """Download/refresh market data in the background so no web request waits on it."""
    STATE["universe"] = uni = load_universe(LIMIT)
    force = False
    while True:
        try:
            STATE["panel"] = get_panel(sorted(uni), refresh=force)
            STATE["error"] = None
            wait = 3 * 3600
        except Exception as e:  # keep serving the old panel, retry later
            STATE["error"] = str(e)
            wait = 300
        STATE["refresh"].wait(wait)
        force = STATE["refresh"].is_set()
        STATE["refresh"].clear()


@app.on_event("startup")
def _start():
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/status")
def status():
    p = STATE["panel"]
    return {"ready": p is not None, "asof": p.asof if p is not None else None,
            "loaded": PROGRESS["done"], "total": PROGRESS["total"], "error": STATE["error"]}


class Position(BaseModel):
    ticker: str
    shares: float
    cost: float = 0


class PortfolioIn(BaseModel):
    cash: float = 0
    positions: list[Position] = []


class ImportIn(BaseModel):
    text: str


class Settings(BaseModel):
    top_n: int = Field(10, ge=3, le=30)
    max_weight: float = Field(0.15, ge=0.05, le=0.5)
    stop_loss: float = Field(0.12, ge=0.03, le=0.5)
    regime_filter: bool = True
    refresh: bool = False


class BacktestIn(Settings):
    years: float = Field(3.0, ge=0.5, le=4.0)
    initial_cash: float = Field(10000, ge=100, le=1e8)
    commission: float = Field(0.001, ge=0, le=0.02)
    slippage: float = Field(0.0005, ge=0, le=0.02)
    rebalance_days: int = Field(5, ge=1, le=63)


def _panel(extra: list[str], refresh: bool):
    if refresh:
        STATE["refresh"].set()
    base = STATE["panel"]
    if base is None:
        if STATE["error"]:
            raise HTTPException(502, f"Could not load market data: {STATE['error']}")
        raise HTTPException(503, f"loading: market data is downloading "
                                 f"({PROGRESS['done']}/{PROGRESS['total']} tickers). This page retries automatically.")
    return STATE["universe"], add_holdings(base, tuple(extra))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/portfolio")
def get_portfolio():
    return portfolio.load()


@app.put("/api/portfolio")
def put_portfolio(body: PortfolioIn):
    try:
        return portfolio.save(body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/portfolio/import")
def import_portfolio(body: ImportIn):
    rows = portfolio.parse_csv(body.text)
    if not rows:
        raise HTTPException(422, "No rows found. Use lines like: AAPL,10,150.25")
    cur = portfolio.load()
    try:
        return portfolio.save({"cash": cur["cash"], "positions": rows})
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/recommendations")
def recommendations(s: Settings):
    pf = portfolio.load()
    uni, panel = _panel([p["ticker"] for p in pf["positions"]], s.refresh)
    params = Params(top_n=s.top_n, max_weight=s.max_weight, stop_loss=s.stop_loss,
                    regime_filter=s.regime_filter)
    # holdings outside the universe may be missing from the panel -> flagged REVIEW
    return recommend(panel, uni, pf, params)


@app.post("/api/backtest")
def backtest_endpoint(b: BacktestIn):
    uni, panel = _panel([], b.refresh)
    params = Params(top_n=b.top_n, max_weight=b.max_weight, stop_loss=b.stop_loss,
                    regime_filter=b.regime_filter, rebalance_days=b.rebalance_days)
    try:
        return run_backtest(panel, params, b.years, b.initial_cash, b.commission, b.slippage)
    except ValueError as e:
        raise HTTPException(422, str(e))
