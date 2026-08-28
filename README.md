# Autonomous Crypto Trading Lab

An autonomous research laboratory for discovering, testing, falsifying and validating crypto trading strategies before producing TradingView Pine Script.

## Design principles
- Research is for discovering hypotheses, not proving them.
- Internet sources are evidence for ideas, never proof of profitability.
- Every candidate must face out-of-sample and robustness/falsification checks.
- The baseline account is **$500 USD**.
- Spot, perpetual futures, long, short, leverage, compounding and multi-strategy portfolios are options for research, not assumptions.
- Backtest results must include realistic fees and slippage assumptions.
- Pine Script must reproduce the validated strategy logic as closely as the platform permits.
- Failed experiments are retained as research knowledge.

## Intended workflow
`DISCOVER → HYPOTHESIZE → BUILD → BACKTEST → ATTACK → VALIDATE → COMPARE → SELECT → PINE → REPEAT`

## Repository layout
- `lab/` — core research framework
- `config/` — central configuration
- `strategies/` — candidate strategy specifications
- `experiments/` — reproducible experiment records
- `reports/` — generated reports
- `pine/` — TradingView output
- `colab/` — minimal Colab launcher
- `.github/workflows/` — automated quality checks

## User experience
The final Colab interface is intentionally minimal. The target is one launch action; data acquisition, experiment orchestration, validation and reporting remain inside the project.

## Important
This repository is a research system, not a promise of profitable trading. Real-money deployment requires independent verification and appropriate risk controls.