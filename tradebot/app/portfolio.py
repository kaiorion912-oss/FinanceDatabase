"""Your portfolio, stored locally as JSON (never sent anywhere)."""

import csv
import io
import json
import os
import re
from pathlib import Path

PATH = Path(os.environ.get("TRADEBOT_PORTFOLIO", Path(__file__).resolve().parents[1] / "data" / "portfolio.json"))
TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def validate(raw: dict) -> dict:
    cash = float(raw.get("cash", 0) or 0)
    if cash < 0:
        raise ValueError("Cash cannot be negative.")
    merged: dict[str, dict] = {}
    for p in raw.get("positions", []):
        t = str(p.get("ticker", "")).strip().upper().replace(".", "-")
        if not t:
            continue
        if not TICKER.match(t):
            raise ValueError(f"Invalid ticker: {t!r}")
        sh, cost = float(p.get("shares", 0) or 0), float(p.get("cost", 0) or 0)
        if sh <= 0 or cost < 0:
            raise ValueError(f"{t}: shares must be > 0 and cost >= 0.")
        if t in merged:  # combine duplicate lines with a weighted-average cost
            m = merged[t]
            tot = m["shares"] + sh
            m["cost"] = (m["cost"] * m["shares"] + cost * sh) / tot
            m["shares"] = tot
        else:
            merged[t] = {"ticker": t, "shares": sh, "cost": cost}
    return {"cash": cash, "positions": list(merged.values())}


def load() -> dict:
    try:
        return validate(json.loads(PATH.read_text()))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {"cash": 0.0, "positions": []}


def save(raw: dict) -> dict:
    data = validate(raw)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(data, indent=2))
    return data


def parse_csv(text: str) -> list[dict]:
    """CSV/paste with columns ticker,shares,cost (header optional; also accepts symbol/quantity/avg_cost)."""
    rows = list(csv.reader(io.StringIO(text.strip())))
    out = []
    for r in rows:
        r = [c.strip() for c in r]
        if len(r) < 2 or not re.match(r"^[A-Za-z]", r[0]) or not re.match(r"^[\d.,]+$", r[1]):
            continue  # header or junk line
        out.append({"ticker": r[0], "shares": r[1].replace(",", ""),
                    "cost": (r[2].replace("$", "").replace(",", "") if len(r) > 2 and r[2] else 0)})
    return out
