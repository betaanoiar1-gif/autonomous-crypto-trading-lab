# Phase 0 Research Foundation

Phase 0 is a gate before strategy discovery.

## Scope

- Spot only
- Long/flat only
- No leverage
- No futures
- No live trading
- No LLM generation
- No arbitrary code execution

## Data target

Default target: 50,000 closed 1h bars for each of 12 liquid USDT spot pairs.
Data is downloaded once and stored as parquet under `experiments/phase0_data_v2/`.

## Integrity gates

1. Monotonic UTC timestamps
2. Unique timestamps
3. Finite OHLCV values
4. Valid OHLC relationships
5. Latest candle is closed
6. Spot position probe must show zero short exposure

## Statistical microscope tests

### Lookahead canary
A deliberately future-dependent signal is run only as a diagnostic. The evaluator must distinguish it materially from flat exposure. Failure blocks discovery because signal plumbing would be suspect.

### Noise floor
Random long/flat signals are evaluated on block-shuffled return paths at doubled trading costs. The false-positive rate must be <= 5% under the Phase 0 acceptance proxy.

### Synthetic edge
A synthetic price process contains a known causal lag-1 edge. A signal using the current bar's return is executed on the next bar, matching the engine's close-to-next-bar semantics. The known edge must remain positive under doubled cost.

## Benchmarks

For representative markets, record:

- buy-and-hold at normal cost;
- buy-and-hold at doubled cost;
- 25% annualized volatility-matched buy-and-hold.

Discovery later reports benchmark-relative alpha rather than raw return alone.

## Phase 0 success

`PHASE0_READY_FOR_DISCOVERY` requires all gates to pass together:

- >= 8 markets loaded;
- >= 5,000 bars per loaded market (default target is 50,000);
- all integrity checks pass;
- spot long/flat enforcement passes;
- lookahead canary passes;
- noise false-positive rate <= 5%;
- synthetic edge detected under normal and doubled cost.

Any failed gate yields `PHASE0_BLOCKED` and discovery must not proceed.

## Burned lockbox datasets

The historical BTC/USDT 1h and ETH/USDT 15m confirmation sets used repeatedly by earlier engine generations are considered research-burned and must not be reused as final lockbox gates.

The final lockbox is specified and hashed before opening and is opened only once per reporting period.
