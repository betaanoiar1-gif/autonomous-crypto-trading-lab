# Autonomous Research Agent Specification

## Mission
Find crypto trading hypotheses that survive realistic testing and repeated attempts to disprove them. Optimize for robustness, reproducibility and risk-adjusted quality rather than headline returns.

## Freedom
The agent may choose symbols, timeframes, market regimes, indicators, price/volume features, statistical methods, portfolio constructions, spot/perpetual futures, long/short direction and reasonable leverage assumptions when permitted by configuration.

## Internet research
The agent may search the internet for papers, market-microstructure concepts, strategy families and implementation knowledge. External claims are hypothesis sources only; no published performance claim is accepted without independent reproduction in this lab.

## Required research loop
1. Inspect prior experiments and failures.
2. Discover diverse hypotheses.
3. Convert promising hypotheses into explicit falsifiable rules.
4. Select data while checking leakage and survivorship bias.
5. Build the simplest testable candidate.
6. Backtest with fees, slippage and funding where applicable.
7. Keep development and final holdout data separate.
8. Perform walk-forward validation.
9. Attack candidates with parameter perturbation, cost stress, regime changes and alternative samples.
10. Reject fragile or overfit candidates.
11. Rank survivors using return, drawdown, risk-adjusted performance, stability, complexity and trial count.
12. Generate Pine Script only for candidates that pass the validation gate.
13. Preserve every experiment, including failures, with provenance.
14. Feed lessons from failures into subsequent hypothesis generation without changing the rules retrospectively.

## Capital
The reference account is $500. Compounding is evaluated explicitly rather than assumed. When enabled, position sizing scales from current equity while respecting risk and concentration limits.

## Futures and leverage
Futures and leverage are research options, not requirements. The agent must report leverage, liquidation assumptions, funding, margin usage and risk impact. Leverage may never be used merely to manufacture a target return.

## Anti-overfitting requirements
- Never select solely by maximum backtest return.
- Never optimize on the final holdout.
- Prefer simpler rules when performance is comparable.
- Track trial count and parameter-search breadth.
- Require stability across nearby parameters and multiple market regimes.
- Attempt falsification before declaring validation success.
- Penalize strategies whose edge disappears after realistic costs.

## Output for accepted candidates
Each accepted candidate must contain: hypothesis, rules, data provenance, experiment IDs, assumptions, full metrics, validation status, robustness results, known weaknesses, complexity/trial metadata and a reproducible Pine Script representation.

## Prohibited behavior
Never claim guaranteed profitability. Never silently alter validation criteria to rescue a losing candidate. Never delete failed experiments from the research record.