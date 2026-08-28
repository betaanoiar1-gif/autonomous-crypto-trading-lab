from __future__ import annotations

from .config import load_settings


def run(agent=None) -> None:
    settings = load_settings()
    print("LAB READY")
    print(f"Initial capital: ${settings.capital.initial_usd:,.2f}")
    if not settings.research.autonomous:
        print("Autonomous research is disabled in configuration.")
        return

    from .research.run import run as research_run

    cycles = max(1, int(settings.research.max_autonomous_cycles))
    target = min(settings.research.max_experiments_per_run, len(research_run.DIVERSITY_SLOTS))
    print(f"Autonomous research cycles: up to {cycles}")
    print(f"Research slots requested per cycle: {target}")

    for cycle in range(1, cycles + 1):
        print(f"\n=== Autonomous cycle {cycle}/{cycles} ===")
        result = research_run(
            max_hypotheses=target,
            agent=agent,
        )
        statuses = [r.get("status", "UNKNOWN") for r in result["records"]]
        validated = [r for r in result["records"] if r.get("status") == "VALIDATED"]
        candidates = [r for r in result["records"] if r.get("status") == "VALIDATION_CANDIDATE"]

        print(f"Cycle run: {result['run_id']}")
        print(f"Hypotheses generated: {result['hypothesis_count']}")
        print(f"Statuses: {statuses}")
        print(f"Validated: {len(validated)} | Candidates: {len(candidates)}")
        print("Results saved under experiments/")

        if validated:
            best = validated[0]
            print("\nAUTONOMOUS VALIDATION ACHIEVED")
            print(f"Selected strategy: {best['hypothesis']['title']}")
            print(f"Symbol: {best['symbol']} | timeframe={best['timeframe']} | market={best['market_type']}")
            if best.get("pine_path"):
                print(f"Pine artifact: {best['pine_path']}")
            print("Research loop stopped after a fully validated candidate was found.")
            return

        print("No fully validated strategy this cycle; continuing autonomously...")

    print("\nAUTONOMOUS RESEARCH LIMIT REACHED")
    print(f"Completed {cycles} research cycles without a VALIDATED strategy.")
    print("No live trading was started.")


if __name__ == "__main__":
    run()
