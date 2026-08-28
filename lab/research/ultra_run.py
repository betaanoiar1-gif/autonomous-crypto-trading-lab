from __future__ import annotations

from . import parallel_run as _engine
from .fast_generator import generate


def run(max_hypotheses: int = 12, agent=None) -> dict:
    _engine._generate_batch = lambda _agent, _snapshot, prior_failures, target: generate(prior_failures, target)
    return _engine.run(max_hypotheses=max_hypotheses, agent=agent)
