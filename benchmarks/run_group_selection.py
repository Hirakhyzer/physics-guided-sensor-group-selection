from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pgsgs import (
    SelectionWeights,
    coupled_process_benchmark,
    evaluate_sensor_dropout,
    select_groups,
)


def main() -> None:
    system, groups = coupled_process_benchmark()
    budgets = [
        {"max_cost": 2.0, "max_energy": 1.6},
        {"max_cost": 3.0, "max_energy": 2.5},
        {"max_cost": 4.0, "max_energy": 3.2},
    ]
    rows = []

    for budget in budgets:
        result = select_groups(system, groups, weights=SelectionWeights(), **budget)
        robustness = evaluate_sensor_dropout(system, result.selected_sensors, failures=1)
        rows.append(
            {
                **budget,
                "selection": asdict(result),
                "dropout": {
                    "baseline_observability": robustness.baseline_observability,
                    "worst_case_observability": robustness.worst_case_observability,
                    "mean_observability": robustness.mean_observability,
                },
            }
        )

    payload = {
        "experiment": "physics-guided-budget-dropout-v1",
        "system": "coupled-process-benchmark",
        "results": rows,
        "note": "Synthetic reduced-order benchmark; results are simulation evidence only.",
    }
    output = Path("artifacts/group-selection-benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
