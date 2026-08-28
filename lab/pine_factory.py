from __future__ import annotations


def _num(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def build_pine(name: str, family: str, params: dict, allow_short: bool = True) -> str:
    family = family.lower().strip()
    short = str(bool(allow_short)).lower()
    if family == "momentum":
        n = int(max(2, min(200, _num(params, "lookback", 20))))
        body = f"""len = input.int({n}, 'Lookback', minval=2, maxval=200)\nscore = close / close[len] - 1\npos = score > 0 ? 1 : ({short} and score < 0 ? -1 : 0)"""
    elif family == "mean_reversion":
        n = int(max(2, min(200, _num(params, "lookback", 40))))
        ze = _num(params, "z_entry", 1.5)
        zx = _num(params, "z_exit", 0.25)
        body = f"""len = input.int({n}, 'Lookback', minval=2, maxval=200)\nzEntry = input.float({ze:.4f}, 'Z entry', minval=0.5, maxval=4)\nzExit = input.float({zx:.4f}, 'Z exit', minval=0, maxval=2)\nmu = ta.sma(close, len)\nsd = ta.stdev(close, len)\nz = sd == 0 ? 0 : (close - mu) / sd\nvar int pos = 0\nif z < -zEntry\n    pos := 1\nelse if {short} and z > zEntry\n    pos := -1\nelse if math.abs(z) < zExit\n    pos := 0"""
    elif family == "breakout":
        n = int(max(2, min(200, _num(params, "lookback", 40))))
        body = f"""len = input.int({n}, 'Lookback', minval=2, maxval=200)\nupper = ta.highest(high[1], len)\nlower = ta.lowest(low[1], len)\nvar int pos = 0\nif close > upper\n    pos := 1\nelse if {short} and close < lower\n    pos := -1"""
    elif family == "moving_average_cross":
        fast = int(max(2, min(100, _num(params, "fast", 10))))
        slow = int(max(fast + 1, min(300, _num(params, "slow", 40))))
        body = f"""fastLen = input.int({fast}, 'Fast', minval=2, maxval=100)\nslowLen = input.int({slow}, 'Slow', minval=3, maxval=300)\nfast = ta.ema(close, fastLen)\nslow = ta.ema(close, slowLen)\nvar int pos = 0\nif ta.crossover(fast, slow)\n    pos := 1\nelse if {short} and ta.crossunder(fast, slow)\n    pos := -1"""
    else:
        raise ValueError(f"Unsupported Pine family: {family}")

    return f"""//@version=6\nstrategy({name!r}, overlay=true, initial_capital=500, pyramiding=0)\n\n{body}\n\nif pos == 1\n    strategy.entry('L', strategy.long)\nelse if pos == -1\n    strategy.entry('S', strategy.short)\nelse\n    strategy.close('L')\n    strategy.close('S')\n"""
