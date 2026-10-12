# Physics-Guided Sensor Group Selection

**Selecting physically meaningful sensor groups for resilient and efficient cyber-physical state estimation.**

This repository studies sensor selection as a **grouped, physics-constrained decision problem** rather than a simple ranking of individual sensors. The framework combines observability, physical coupling, redundancy, uncertainty, sensing cost, and failure resilience to identify sensor groups that retain useful system information under budget and fault constraints.

## Research question

> Can physics-guided sensor grouping improve state-estimation quality and resilience compared with selecting sensors independently from data-driven importance scores alone?

## Core idea

```text
Physical system model
        ↓
Candidate sensors + costs
        ↓
Physics graph / coupling matrix
        ↓
Sensor-group construction
        ↓
Observability + redundancy analysis
        ↓
Budget / energy / failure constraints
        ↓
Group-selection optimizer
        ↓
Selected sensor groups
        ↓
State-estimation evaluation
        ↓
Robustness / ablation / transfer studies
```

## Initial capabilities

- explicit sensor and sensor-group data models;
- physics-coupling graph representation;
- group observability scoring;
- redundancy and failure-tolerance scoring;
- sensing-cost and energy-aware constraints;
- greedy multi-objective group selection;
- sensor-dropout robustness evaluation;
- synthetic benchmark systems;
- deterministic experiments and tests.

## Why group selection?

Many CPS variables are not independently informative. A pressure sensor may only be useful when interpreted with flow and level; a current sensor may be valuable only together with voltage and temperature. Independent sensor ranking can therefore choose individually strong but collectively redundant sensors.

This project instead evaluates **sets of sensors that have physical meaning together**.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python examples/run_demo.py
python benchmarks/run_group_selection.py
```

## Research directions

1. physics-guided vs data-only sensor selection;
2. observability under sensor dropout;
3. adversarial or faulty-sensor resilience;
4. energy-aware sensing budgets;
5. group sparsity and redundancy trade-offs;
6. cross-domain transfer of grouping strategies;
7. adaptive group reconfiguration as operating conditions change.

## Scientific boundary

The included systems are synthetic reduced-order research models. Results should be interpreted as controlled simulation evidence, not deployment guarantees.

## License

MIT
