FROM python:3.12-slim
WORKDIR /srv
COPY tradebot/requirements.txt tradebot/requirements.txt
RUN pip install --no-cache-dir -r tradebot/requirements.txt
COPY database/equities database/equities
COPY tradebot tradebot
WORKDIR /srv/tradebot
ENV HOST=0.0.0.0 TRADEBOT_PORTFOLIO=/data/portfolio.json TRADEBOT_CACHE=/data/cache
CMD ["python", "-m", "app"]
