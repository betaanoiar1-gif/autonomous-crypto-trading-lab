from __future__ import annotations


def _num(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def build_pine(name: str, family: str, params: dict, allow_short: bool = True) -> str:
    family = family.lower().strip()
    if family == "momentum":
        n = int(max(2, min(200, _num(params, "lookback", 20))))
        body = f"""len = input.int({n}, 'Lookback', minval=2, maxval=200)\nret = close / close[len] - 1\nlongCond = ret > 0\nshortCond = ret < 0\nif longCond\n    strategy.entry('L', strategy.long)\nif {str(allow_short).lower()} and shortCond\n    strategy.entry('S', strategy.short)\nif not longCond and not shortCond\n    strategy.close('L')\n    strategy.close('S')"""
    elif family == "mean_reversion":
        n = int(max(2, min(200, _num(params, "lookback", 40))))
        ze = _num(params, "z_entry", 1.5)
        zx = _num(params, "z_exit", 0.25)
        body = f"""len = input.int({n}, 'Lookback', minval=2, maxval=200)\nzEntry = input.float({ze:.4f}, 'Z entry', minval=0.5, maxval=4)\nzExit = input.float({zx:.4f}, 'Z exit', minval=0, maxval=2)\nmu = ta.sma(close, len)\nsd = ta.stdev(close, len)\nz = sd == 0 ? 0 : (close - mu) / sd\nif z < -zEntry\n    strategy.entry('L', strategy.long)\nif {str(allow_short).lower()} and z > zEntry\n    strategy.entry('S', strategy.short)\nif math.abs(z) < zExit\n    strategy.close('L')\n    strategy.close('S')"""
    elif family == "breakout":
        n = int(max(2, min(200, _num(params, "lookback", 40))))
        body = f"""len = input.int({n}, 'Lookback', minval=2, maxval=200)\nupper = ta.highest(high[1], len)\nlower = ta.lowest(low[1], len)\nif close > upper\n    strategy.entry('L', strategy.long)\nif {str(allow_short).lower()} and close < lower\n    strategy.entry('S', strategy.short)\n"""
    elif family == "moving_average_cross":
        fast = int(max(2, min(100, _num(params, "fast", 10))))
        slow = int(max(fast + 1, min(300, _num(params, "slow", 40))))
        body = f"""fastLen = input.int({fast}, 'Fast', minval=2, maxval=100)\nslowLen = input.int({slow}, 'Slow', minval=3, maxval=300)\nfast = ta.ema(close, fastLen)\nslow = ta.ema(close, slowLen)\nif ta.crossover(fast, slow)\n    strategy.entry('L', strategy.long)\nif {str(allow_short).lower()} and ta.crossunder(fast, slow)\n    strategy.entry('S', strategy.short)"""
    else:
        raise ValueError(f"Unsupported Pine family: {family}")
    return f"//@version=6\nstrategy({name!r}, overlay=true, initial_capital=500, pyramiding=0)\n\n{body}\n"