from dataclasses import dataclass

@dataclass
class PineSpec:
    name: str
    version: str
    body: str


def render(spec: PineSpec) -> str:
    return (
        f'//@version=6\n'
        f'strategy("{spec.name} {spec.version}", overlay=true, initial_capital=500, pyramiding=0)\n\n'
        + spec.body.strip() + "\n"
    )
