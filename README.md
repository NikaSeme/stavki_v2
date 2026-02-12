# STAVKI V2 — Sports Betting Analytics System

A comprehensive ML-powered sports betting analytics platform for finding value bets.

## 🎯 Overview

STAVKI analyzes betting markets using ensemble machine learning models to identify positive expected value (EV) opportunities. The system combines:

- **Multiple ML Models**: Poisson, CatBoost, LightGBM, Neural Networks
- **Smart Blending**: Adjusts model/market trust based on league efficiency  
- **Kelly Staking**: Optimal bankroll management with risk controls
- **Backtesting**: Monte Carlo, Walk-Forward validation, reality simulation

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourname/stavki_v2.git
cd stavki_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## 🚀 Quick Start

### Command Line

```bash
# Find value bets
stavki predict --league soccer_epl --min-ev 0.05

# Run backtest
stavki backtest --data data/historical.csv --kelly 0.25

# Train models
stavki train --data data/features.csv --epochs 100

# Check status
stavki status
```

### Python API

```python
from stavki.pipelines import DailyPipeline, PipelineConfig
from stavki.backtesting import BacktestEngine, BacktestConfig

# Find value bets
config = PipelineConfig(leagues=["soccer_epl"], min_ev=0.03)
pipeline = DailyPipeline(config=config, bankroll=1000)
bets = pipeline.run()

for bet in bets:
    print(f"{bet.selection} @ {bet.odds:.2f} | EV: {bet.ev:.1%}")

# Run backtest
import pandas as pd
data = pd.read_csv("data/historical.csv")

config = BacktestConfig(min_ev=0.05, kelly_fraction=0.25)
engine = BacktestEngine(config=config)
result = engine.run(data)

print(f"ROI: {result.roi:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
```

## 📂 Project Structure

```
stavki_v2/
├── stavki/
│   ├── data/           # Data collectors & processors
│   │   ├── collectors/ # OddsAPI, SportMonks, Weather
│   │   ├── processors/ # Data cleaning & normalization
│   │   └── storage/    # Result caching
│   │
│   ├── features/       # Feature engineering
│   │   ├── builders/   # ELO, form, odds, disagreement
│   │   └── store.py    # Feature management
│   │
│   ├── models/         # Prediction models
│   │   ├── poisson/    # Dixon-Coles Poisson model
│   │   ├── catboost/   # CatBoost gradient boosting
│   │   ├── lightgbm/   # LightGBM
│   │   ├── neural/     # Multi-task neural network
│   │   └── ensemble/   # Model ensembling
│   │
│   ├── strategy/       # Betting strategy
│   │   ├── ev.py       # EV calculation, vig removal
│   │   ├── kelly.py    # Kelly staking, risk mgmt
│   │   ├── filters.py  # Bet filtering, meta-filter
│   │   └── optimizer.py # Weight/threshold optimization
│   │
│   ├── pipelines/      # End-to-end pipelines
│   │   ├── daily.py    # Daily betting pipeline
│   │   └── training.py # Model training pipeline
│   │
│   ├── backtesting/    # Backtesting infrastructure
│   │   ├── engine.py   # Backtest engine
│   │   └── metrics.py  # Performance metrics
│   │
│   └── interfaces/     # User interfaces
│       ├── cli.py      # Command line interface
│       ├── telegram_bot.py
│       └── scheduler.py
│
├── config/             # Configuration files
├── data/               # Data storage
├── models/             # Trained model checkpoints
└── outputs/            # Pipeline outputs & reports
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# API Keys
ODDS_API_KEY=your_key_here
SPORTMONKS_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Settings
DEFAULT_BANKROLL=1000
MIN_EV_THRESHOLD=0.03
KELLY_FRACTION=0.25
```

### League Configuration

Edit `config/league_config.json`:

```json
{
  "EPL": {
    "tier": "tier1",
    "policy": "BET",
    "weights": {"catboost": 0.35, "neural": 0.35, "poisson": 0.30}
  },
  "LaLiga": {
    "tier": "tier1",
    "policy": "BET"
  }
}
```

## 📊 Models

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Poisson** | Dixon-Coles goal model | Interpretable, handles low-scoring |
| **CatBoost** | Gradient boosting | Feature importance, handles categories |
| **LightGBM** | Light gradient boosting | Fast training, efficient |
| **Neural** | Multi-task network | Complex patterns, multi-output |

## 🧪 Backtesting

```python
from stavki.backtesting import BacktestEngine, MonteCarloSimulator

# Standard backtest
result = engine.run(data)

# Monte Carlo confidence intervals
mc = MonteCarloSimulator(n_simulations=10000)
mc_result = mc.simulate(result)
print(f"95% CI: [{mc_result['roi_ci_lower']:.2%}, {mc_result['roi_ci_upper']:.2%}]")
```

### Reality Scenarios

```python
from stavki.backtesting import RealitySimulator

# Test under different conditions
for scenario in ["optimistic", "realistic", "pessimistic", "worst_case"]:
    sim = RealitySimulator(scenario=scenario)
    config = sim.adjust_config(base_config)
    result = BacktestEngine(config).run(data)
    print(f"{scenario}: ROI={result.roi:.2%}")
```

## 📈 Key Metrics

- **ROI**: Return on Investment
- **Sharpe Ratio**: Risk-adjusted return  
- **CLV**: Closing Line Value (benchmark vs closing odds)
- **Max Drawdown**: Largest peak-to-trough decline

## 🔐 Risk Management

1. **Fractional Kelly**: Uses 25% of full Kelly stake
2. **Exposure Limits**: Max 5% per bet, 20% per league
3. **Drawdown Protection**: Reduces stakes during drawdowns
4. **Meta-Filter**: Requires model agreement before betting

## 📱 Telegram Bot

```bash
# Set token in .env
TELEGRAM_BOT_TOKEN=your_token

# Run bot
python -m stavki.interfaces.telegram_bot
```

Commands:
- `/bets` — Current value bets
- `/status` — System status
- `/subscribe` — Get alerts
- `/help` — Help

## 🛠️ Development

```bash
# Run tests
pytest tests/

# Type checking
mypy stavki/

# Linting
ruff check stavki/
```

## 📝 License

MIT License — see LICENSE file.

## ⚠️ Disclaimer

This software is for educational purposes only. Sports betting involves financial risk. Past performance does not guarantee future results. Always bet responsibly.
