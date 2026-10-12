from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pgsgs.baselines import cheapest_first, observability_greedy, reliability_first
from pgsgs.benchmarks import benchmark_suite
from pgsgs.evaluation import evaluate_sensor_set
from pgsgs.scenarios import DisturbanceKind, SensorDisturbance
from pgsgs.selection import select_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain physics-guided sensor selection study")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", default="7,11,19,23,31")
    parser.add_argument("--max-cost", type=float, default=3.0)
    parser.add_argument("--max-energy", type=float, default=2.4)
    parser.add_argument("--output", default="artifacts/cross-domain.json")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]

    rows = []
    selections = {}
    for domain, (system, groups) in benchmark_suite().items():
        target = tuple(system.sensors)[1]
        start, end = args.steps // 3, 2 * args.steps // 3
        scenarios = {
            "clean": (),
            "dropout": (SensorDisturbance(target, DisturbanceKind.DROPOUT, start, end),),
            "bias": (SensorDisturbance(target, DisturbanceKind.BIAS, start, end, 0.6),),
            "drift": (SensorDisturbance(target, DisturbanceKind.DRIFT, start, end, 0.8),),
            "freeze": (SensorDisturbance(target, DisturbanceKind.FREEZE, start, end),),
        }
        physics = select_groups(system, groups, max_cost=args.max_cost, max_energy=args.max_energy)
        methods = {
            "physics-guided-groups": physics.selected_sensors,
            "observability-greedy": observability_greedy(system, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
            "reliability-first": reliability_first(system, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
            "cheapest-first": cheapest_first(system, max_cost=args.max_cost, max_energy=args.max_energy).sensors,
        }
        selections[domain] = {name: list(value) for name, value in methods.items()}
        for seed in seeds:
            for method, sensors in methods.items():
                summary = evaluate_sensor_set(system, sensors, scenarios, method=method, steps=args.steps, seed=seed)
                rows.append({"domain": domain, "seed": seed, **asdict(summary)})

    payload = {
        "protocol": "cross-domain-physics-guided-v1",
        "steps": args.steps,
        "seeds": seeds,
        "budgets": {"cost": args.max_cost, "energy": args.max_energy},
        "selections": selections,
        "runs": rows,
        "boundary": "All domains are synthetic reduced-order research models.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
