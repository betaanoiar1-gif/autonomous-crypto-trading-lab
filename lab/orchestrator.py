from __future__ import annotations

from .config import load_settings


def run(agent=None) -> None:
    settings = load_settings()
    print("LAB READY")
    print(f"Initial capital: ${settings.capital.initial_usd:,.2f}")
    if settings.research.autonomous:
        print("Starting autonomous research cycle...")
        from .research.run import run as research_run
        result = research_run(
            max_hypotheses=min(4, settings.research.max_experiments_per_run),
            agent=agent,
        )
        statuses = [r.get("status", "UNKNOWN") for r in result["records"]]
        print(f"Research run: {result['run_id']}")
        print(f"Hypotheses generated: {result['hypothesis_count']}")
        print(f"Statuses: {statuses}")
        print("Results saved under experiments/")
    else:
        print("Autonomous research is disabled in configuration.")


if __name__ == "__main__":
    run()
