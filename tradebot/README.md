# Trade Suggestion Dashboard

A local dashboard that scores every NYSE/Nasdaq stock (~7,400; illiquid names filtered out) on real daily market data
(Yahoo Finance), suggests BUY / SELL / TRIM / HOLD trades for **your** portfolio, and
replays the same rules with paper money to backtest them.

Built on the vendored `finance` toolkit (`vendor/finance`, MIT, from shashankvemuri/Finance:
indicators + backtest engine). The stock universe comes from this repo's `database/equities` CSVs.

## Run
```bash
cd tradebot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app            # open http://127.0.0.1:8000
```
First request downloads ~5y of prices for ~7,400 tickers in batches (10+ minutes; cached 6h in `.cache/`). Stocks under $5 or under $5M/day traded are never picked. Offline UI demo with
clearly-labelled fake prices: `TRADEBOT_DEMO=1 python -m app`.

## Tabs
- **Today's picks**: ranked suggestions + top-25 leaderboard, sized to your cash/equity.
- **My portfolio**: enter holdings (ticker, shares, avg cost, cash) or paste/upload CSV. Stored in `data/portfolio.json`.
- **Paper-money backtest**: virtual cash account, next-open fills, commission + slippage, vs SPY buy & hold.

## The model
Score = cross-sectional rank blend of 12-1m momentum, 6m and 3m momentum, relative strength vs SPY,
low volatility, MACD. Eligible only if price > 200d avg, 50d > 200d, 12-1m momentum > 0.
Top-N are held (inverse-volatility weights, capped per position); holdings are kept while rank
stays in the top 2N. SELL on broken trend, rank collapse, or stop-loss vs your cost. Exposure is
halved when SPY is below its 200d average. Suggested stop = price - 2.5xATR.

## Caveats
Educational tool, not investment advice. Backtests suffer survivorship bias (today's universe)
and ignore taxes. Data is daily, end-of-day.

Tests: `pytest tests -q` (synthetic data).

## Host it (public link)
Needs a host that can reach Yahoo Finance and run a Docker container. `render.yaml` + `Dockerfile`
at the repo root deploy to Render: New → Blueprint → pick this repo/branch → set `TRADEBOT_PASSWORD`.
The page then asks for a password (any username). Prices download in the background after the
server starts; the dashboard shows progress and retries. Without a persistent disk, your saved
portfolio is lost on each redeploy. Env vars: `TRADEBOT_PASSWORD`, `TRADEBOT_UNIVERSE_LIMIT`
(fewer stocks = less RAM), `PORT`, `HOST`.
