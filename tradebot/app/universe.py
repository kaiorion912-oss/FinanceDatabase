"""Stock universe: US-listed mega/large-cap equities from this repo's FinanceDatabase CSVs."""

import csv
import re
from pathlib import Path

DB_DIR = Path(__file__).resolve().parents[2] / "database" / "equities"
SYMBOL = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")

# Used only if the database folder is missing (e.g. app copied out of the repo).
FALLBACK = {
    "AAPL": ("Apple", "Information Technology"), "MSFT": ("Microsoft", "Information Technology"),
    "NVDA": ("NVIDIA", "Information Technology"), "GOOGL": ("Alphabet", "Communication Services"),
    "AMZN": ("Amazon", "Consumer Discretionary"), "META": ("Meta Platforms", "Communication Services"),
    "TSLA": ("Tesla", "Consumer Discretionary"), "AVGO": ("Broadcom", "Information Technology"),
    "JPM": ("JPMorgan Chase", "Financials"), "V": ("Visa", "Financials"),
    "UNH": ("UnitedHealth", "Health Care"), "XOM": ("Exxon Mobil", "Energy"),
    "LLY": ("Eli Lilly", "Health Care"), "WMT": ("Walmart", "Consumer Staples"),
    "COST": ("Costco", "Consumer Staples"), "HD": ("Home Depot", "Consumer Discretionary"),
}


US_MICS = {"XNYS", "XNAS"}  # the foreign-exchange listings of US firms have unusable tickers


def load_universe(limit: int = 300) -> dict[str, dict]:
    """Return {symbol: {name, sector}}: non-delisted US-listed Mega Cap, then Large Cap, equities."""
    out: dict[str, dict] = {}
    if DB_DIR.is_dir():
        for tier in ("Mega Cap", "Large Cap"):
            for path in sorted(DB_DIR.glob("*.csv")):
                with path.open(newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        sym = row["symbol"].strip().upper().replace(".", "-")
                        if (
                            row["country"] == "United States"
                            and row["mic"] in US_MICS
                            and row["market_cap"] == tier
                            and row["delisted"] != "True"
                            and SYMBOL.match(sym)
                            and sym not in out
                        ):
                            out[sym] = {"name": row["name"], "sector": row["sector"] or "Unknown"}
    if not out:
        out = {s: {"name": n, "sector": sec} for s, (n, sec) in FALLBACK.items()}
    return dict(list(out.items())[:limit])
