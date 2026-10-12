from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pgsgs.baselines import cheapest_first, individual_group_baseline, observability_greedy, reliability_first
from pgsgs.benchmarks import coupled_process_benchmark
from pgsgs.evaluation import evaluate_sensor_set
from pgsgs.scenarios import DisturbanceKind, SensorDisturbance
from pgsgs.selection import select_groups


def scenarios(steps: int):
    start, end = steps // 3, 2 * steps // 3
    return {
        "clean": (),
        "flow-dropout": (SensorDisturbance("flow", DisturbanceKind.DROPOUT, start, end),),
        "flow-bias": (SensorDisturbance("flow", DisturbanceKind.BIAS, start, end, 0.6),),
        "flow-drift": (SensorDisturbance("flow", DisturbanceKind.DRIFT, start, end, 0.8),),
        "flow-freeze": (SensorDisturbance("flow", DisturbanceKind.FREEZE, start, end),),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full physics-guided sensor selection benchmark")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", default="7,11,19,23,31")
    parser.add_argument("--max-cost", type=float, default=3.2)
    parser.add_argument("--max-energy", type=float, default=2.6)
    parser.add_argument("--output", default="artifacts/full-evaluation.json")
    args = parser.parse_args()

    system, groups = coupled_process_benchmark()
    physics = select_groups(system, groups, max_cost=args.max_cost, max_energy=args.max_energy)
    methods = {
        "physics-guided-groups": physics.selected_sensors,
        "observability-greedy": observability_greedy(system, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
        "reliability-first": reliability_first(system, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
        "cheapest-first": cheapest_first(system, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
        "membership-frequency": individual_group_baseline(system, groups, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
    }

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    runs = []
    for seed in seeds:
        for method, sensor_names in methods.items():
            summary = evaluate_sensor_set(
                system,
                sensor_names,
                scenarios(args.steps),
                method=method,
                steps=args.steps,
                seed=seed,
            )
            runs.append({"seed": seed, **asdict(summary)})

    payload = {
        "protocol": "physics-guided-selection-v1",
        "steps": args.steps,
        "seeds": seeds,
        "budgets": {"cost": args.max_cost, "energy": args.max_energy},
        "selection": {
            "groups": physics.selected_groups,
            "sensors": physics.selected_sensors,
            "cost": physics.total_cost,
            "energy": physics.total_energy,
            "utility": physics.utility,
            "marginal_trace": physics.marginal_trace,
        },
        "methods": {name: list(sensors) for name, sensors in methods.items()},
        "runs": runs,
        "interpretation": "Synthetic reduced-order benchmark; results are not deployment guarantees.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
