import os

os.environ["TRADEBOT_DEMO"] = "1"

import pytest
from fastapi.testclient import TestClient

import app.portfolio as pf
from app.backtest import run_backtest
from app.data import get_panel
from app.server import app
from app.strategy import Params, compute_factors, recommend, target_weights
from app.universe import load_universe


@pytest.fixture(scope="module")
def panel():
    return get_panel(list(load_universe())[:80], years=5)


def test_universe_us_listings():
    u = load_universe()
    assert len(u) > 100 and "AAPL" in u and not any(s.endswith("-F") for s in u)


def test_targets_valid_and_no_lookahead(panel):
    p = Params()
    f = compute_factors(panel, p)
    w = target_weights(f, panel, p, start=260)
    assert (w.sum(axis=1) <= 1 + 1e-9).all() and (w >= 0).all().all() and (w.max(axis=1) <= p.max_weight + 1e-9).all()
    # truncating the future must not change past targets (no look-ahead)
    cut = panel.close.index[-100]
    short = type(panel)(*(getattr(panel, k).loc[:cut] for k in ("open", "high", "low", "close", "volume")),
                        panel.source, True)
    w2 = target_weights(compute_factors(short, p), short, p, start=260)
    assert (w2 - w.loc[:cut]).abs().max().max() < 1e-9


def test_recommend_sizes_within_cash(panel):
    u = load_universe()
    r = recommend(panel, u, {"cash": 5000, "positions": []})
    spent = sum(x["shares"] * x["price"] for x in r["recommendations"] if x["action"] == "BUY")
    assert spent <= 5000 + 1e-6
    assert all(x["action"] == "BUY" for x in r["recommendations"])


def test_recommend_flags_unknown_holding(panel):
    r = recommend(panel, {}, {"cash": 0, "positions": [{"ticker": "ZZZZ", "shares": 1, "cost": 1}]})
    assert r["recommendations"][0]["action"] == "REVIEW"


def test_backtest_accounting(panel):
    b = run_backtest(panel, Params(), years=2, initial_cash=10000)
    assert b["final_equity"] > 0 and len(b["curve"]["dates"]) == len(b["curve"]["equity"])
    assert b["metrics"]["max_drawdown"] <= 0


def test_portfolio_validation_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "PATH", tmp_path / "p.json")
    d = pf.save({"cash": 10, "positions": [{"ticker": "aapl", "shares": 2, "cost": 100}, {"ticker": "AAPL", "shares": 2, "cost": 200}]})
    assert d["positions"] == [{"ticker": "AAPL", "shares": 4.0, "cost": 150.0}]
    with pytest.raises(ValueError):
        pf.save({"positions": [{"ticker": "bad!!", "shares": 1}]})
    assert pf.parse_csv("ticker,shares,cost\nMSFT,5,310\nBRK.B,1,\n")[0]["ticker"] == "MSFT"


def test_api(tmp_path, monkeypatch, panel):
    import app.server as srv
    monkeypatch.setattr(pf, "PATH", tmp_path / "p.json")
    monkeypatch.setitem(srv.STATE, "panel", None)
    c = TestClient(app)
    assert c.post("/api/recommendations", json={}).status_code == 503  # still loading
    assert c.get("/api/status").json()["ready"] is False
    monkeypatch.setitem(srv.STATE, "panel", panel)
    monkeypatch.setitem(srv.STATE, "universe", load_universe())
    assert c.put("/api/portfolio", json={"cash": 100, "positions": [{"ticker": "AAPL", "shares": 1, "cost": 1}]}).status_code == 200
    assert c.put("/api/portfolio", json={"cash": -1}).status_code == 422
    assert c.post("/api/recommendations", json={}).json()["synthetic"] is True
    assert c.post("/api/backtest", json={"years": 1}).status_code == 200
    assert c.get("/").status_code == 200


def test_password_gate(monkeypatch):
    import app.server as srv
    monkeypatch.setattr(srv, "PASSWORD", "s3cret")
    c = TestClient(app)
    assert c.get("/").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert c.get("/", auth=("me", "wrong")).status_code == 401
    assert c.get("/", auth=("me", "s3cret")).status_code == 200


def test_full_universe_and_liquidity_guard():
    u = load_universe()
    assert len(u) > 5000 and "AAPL" in u
    import numpy as np
    from app.data import Panel, _synthetic
    pan = _synthetic(["AAA", "BBB", "SPY"], 5)
    pan.volume["BBB"] = 10.0  # ~$1k/day: illiquid, must never be eligible
    f = compute_factors(pan, Params())
    assert "BBB" not in f.eligible.columns  # illiquid names are never scored
    r = recommend(pan, {}, {"cash": 0, "positions": [{"ticker": "BBB", "shares": 1, "cost": 1}]})
    assert r["recommendations"][0]["action"] == "REVIEW"


def test_vectorised_indicators_match_toolkit():
    from app.data import _synthetic
    from app.strategy import atr_panel, macd_hist_panel, rsi_panel
    from finance.indicators import atr, macd, rsi

    pan = _synthetic(["AAA", "SPY"], 5)
    c, h, l = pan.close[["AAA"]], pan.high[["AAA"]], pan.low[["AAA"]]
    assert abs(rsi_panel(c)["AAA"].iloc[-1] - rsi(c["AAA"]).iloc[-1]) < 0.5
    assert abs(atr_panel(h, l, c)["AAA"].iloc[-1] / atr(h["AAA"], l["AAA"], c["AAA"]).iloc[-1] - 1) < 0.01
    assert abs(macd_hist_panel(c)["AAA"].iloc[-1] - macd(c["AAA"])["histogram"].iloc[-1]) < 1e-6
