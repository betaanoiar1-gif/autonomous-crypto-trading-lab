from __future__ import annotations

from .config import load_settings


def run(agent=None) -> None:
    settings = load_settings()
    print("LAB READY", flush=True)
    print(f"Initial capital: ${settings.capital.initial_usd:,.2f}", flush=True)
    if not settings.research.autonomous:
        print("Autonomous research is disabled in configuration.", flush=True)
        return

    from .research.run import DIVERSITY_SLOTS
    from .research.ultra_fast_run import run as research_run

    cycles = max(1, int(getattr(settings.research, "max_autonomous_cycles", 10)))
    target = min(int(settings.research.max_experiments_per_run), len(DIVERSITY_SLOTS))
    print(f"Autonomous research cycles: up to {cycles}", flush=True)
    print(f"Research slots requested per cycle: {target}", flush=True)

    for cycle in range(1, cycles + 1):
        print(f"\n=== Autonomous cycle {cycle}/{cycles} ===", flush=True)
        result = research_run(max_hypotheses=target, agent=agent)
        statuses = [r.get("status", "UNKNOWN") for r in result["records"]]
        validated = [r for r in result["records"] if r.get("status") == "VALIDATED"]
        candidates = [r for r in result["records"] if r.get("status") == "VALIDATION_CANDIDATE"]

        print(f"Cycle run: {result['run_id']}", flush=True)
        print(f"Hypotheses generated: {result['hypothesis_count']}", flush=True)
        print(f"Statuses: {statuses}", flush=True)
        print(f"Validated: {len(validated)} | Candidates: {len(candidates)}", flush=True)
        print("Results saved under experiments/", flush=True)

        if validated:
            best = validated[0]
            print("\nAUTONOMOUS VALIDATION ACHIEVED", flush=True)
            print(f"Selected strategy: {best['hypothesis']['title']}", flush=True)
            print(f"Symbol: {best['symbol']} | timeframe={best['timeframe']} | market={best['market_type']}", flush=True)
            if best.get("pine_path"):
                print(f"Pine artifact: {best['pine_path']}", flush=True)
            print("Research loop stopped after a fully validated candidate was found.", flush=True)
            return

        print("No fully validated strategy this cycle; continuing autonomously...", flush=True)

    print("\nAUTONOMOUS RESEARCH LIMIT REACHED", flush=True)
    print(f"Completed {cycles} research cycles without a VALIDATED strategy.", flush=True)
    print("No live trading was started.", flush=True)


if __name__ == "__main__":
    run()
