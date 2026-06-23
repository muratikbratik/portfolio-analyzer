# Portfolio Risk Analyzer

AI-powered portfolio risk analysis platform with Monte Carlo simulation and real market data.

## Live Demo
🔗 [Open App](https://portfolio-analyzer-hvediz9iuntehwijo3bpfa.streamlit.app/)

## Features
- **Real market data** via Yahoo Finance (yfinance)
- **Key risk metrics** — Sharpe Ratio, VaR (95%), Max Drawdown, Beta, Annual Return
- **Portfolio vs S&P 500** — benchmark comparison with live data
- **Monte Carlo Simulation** — 500 paths, 1-year forecast with percentile bands
- **Return Distribution** — histogram with VaR threshold
- **AI Portfolio Analysis** — automated plain-English risk commentary

## Tech Stack
- Python, pandas, numpy, scipy
- Streamlit (interactive UI)
- Plotly (charts)
- yfinance (real market data)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app/main.py
```
