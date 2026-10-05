"""FastAPI server: `python -m app` -> http://127.0.0.1:8000"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import portfolio
from .backtest import run_backtest
from .data import DataError, get_panel
from .strategy import Params, recommend
from .universe import load_universe

STATIC = Path(__file__).resolve().parents[1] / "static"
app = FastAPI(title="Trade Suggestion Dashboard")


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


def _panel(extra: list[str], refresh: bool, years: float = 5.0):
    uni = load_universe()
    try:
        return uni, get_panel(sorted(uni), years=years, refresh=refresh, keep=tuple(extra))
    except DataError as e:
        raise HTTPException(502, str(e))
    except Exception as e:  # network / provider failures surface as a readable message
        raise HTTPException(502, f"Could not load market data: {e}")


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
