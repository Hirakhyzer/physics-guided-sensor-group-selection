# Physics-Guided Sensor Group Selection

[![CI](https://github.com/Hirakhyzer/physics-guided-sensor-group-selection/actions/workflows/ci.yml/badge.svg)](https://github.com/Hirakhyzer/physics-guided-sensor-group-selection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Research](https://img.shields.io/badge/focus-resilient%20CPS-orange)

**Physics-guided, budget-aware sensor group selection for resilient cyber-physical state estimation.**

Instead of ranking sensors independently, this project treats sensing as a structured decision problem: sensors belong to physically meaningful groups, groups overlap, resources are finite, failures occur, and the final objective is accurate state estimation rather than a high internal selection score.

## Research question

> **Can physics-guided sensor groups preserve state-estimation quality and resilience under sensing cost, energy, fault, and observability constraints better than independent sensor-selection baselines?**

## What is implemented

- typed sensor, sensor-group, and linear CPS models;
- observability matrix/rank analysis;
- physics-coupling, redundancy, reliability, cost, and energy metrics;
- **marginal overlap-aware group selection** under cost/energy budgets;
- sensor-dropout observability analysis;
- linear Kalman state estimation with missing-measurement support;
- innovation-gated resilient estimation;
- dropout, bias, drift, and freeze telemetry scenarios;
- online reconfiguration after sensor loss or reliability degradation;
- cheapest-first, reliability-first, observability-greedy, and membership-frequency baselines;
- hydraulic-process, thermal-building, and battery electro-thermal benchmark domains;
- paired bootstrap comparison for matched experimental runs;
- JSON experiment artifacts and a machine-readable result schema;
- CLI, tests, CI, citation metadata, and reproducibility protocol.

## Architecture

```text
System dynamics + physics coupling
             |
Sensors + candidate physical groups
             |
Observability / coupling / redundancy / reliability
             |
Marginal group selection <--- cost + energy budgets
             |
Selected sensor union
       +-----+-----+
       |           |
   Kalman      Innovation-gated
 estimator       estimator
       |           |
       +-----+-----+
             |
 dropout / bias / drift / freeze
             |
 estimation + robustness metrics
             |
 degraded sensor health
             |
 adaptive group reconfiguration
             |
 paired cross-domain evaluation
```

See `docs/architecture.md` for the design rationale.

## Quick start

```bash
git clone https://github.com/Hirakhyzer/physics-guided-sensor-group-selection.git
cd physics-guided-sensor-group-selection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
```

Select groups directly from the command line:

```bash
pgsgs select --domain coupled-process --max-cost 3.2 --max-energy 2.6
```

Reconfigure after a sensor becomes unavailable:

```bash
pgsgs adapt \
  --domain coupled-process \
  --unavailable flow \
  --max-cost 3.2 \
  --max-energy 2.6 \
  --minimum-observability 0.75
```

## Reproducible experiments

Run a detailed single-domain comparison:

```bash
python benchmarks/run_full_evaluation.py \
  --steps 120 \
  --seeds 7,11,19,23,31 \
  --output artifacts/full-evaluation.json
```

Run the cross-domain study:

```bash
python benchmarks/run_cross_domain.py \
  --steps 120 \
  --seeds 7,11,19,23,31 \
  --output artifacts/cross-domain.json
```

Both runners use matched seeds and fixed budgets. The cross-domain experiment compares physics-guided groups with non-physics baselines under clean sensing, dropout, bias, drift, and freeze conditions.

## Benchmark domains

| Domain | State interpretation | Example groups |
|---|---|---|
| Coupled process | pressure / flow / level chain | hydraulic core, redundant flow, boundary pressure |
| Thermal building | zones / supply / return thermal state | zone comfort, HVAC loop, thermal chain |
| Battery pack | SOC / voltage / current / temperature | electrical core, electro-thermal, safety monitor |

The domains are intentionally reduced-order and transparent. They are research benchmarks, not validated operational digital twins.

## Evaluation philosophy

A selected set is not considered good merely because the selector assigns it high utility. The project evaluates downstream consequences:

- clean and disturbed state-estimation RMSE;
- worst-case scenario RMSE;
- observability retained after dropout;
- sensing cost and energy;
- resilient-estimator behavior;
- adaptive reconfiguration success;
- matched candidate-vs-baseline deltas and win rate.

`src/pgsgs/statistics.py` provides paired bootstrap intervals. Repeated time steps from one trajectory are **not** treated as independent replicates.

## Project structure

```text
src/pgsgs/
├── model.py                 # sensors, groups, system model
├── physics.py               # observability and physics metrics
├── selection.py             # marginal group optimizer
├── robustness.py            # structural dropout analysis
├── estimation.py            # Kalman state estimator
├── resilient_estimation.py  # innovation-gated estimator
├── scenarios.py             # dropout/bias/drift/freeze simulation
├── adaptive.py              # online reconfiguration
├── baselines.py             # comparison selectors
├── evaluation.py            # end-to-end RMSE evaluation
├── statistics.py            # paired bootstrap comparison
├── benchmarks.py            # three benchmark domains
└── cli.py                   # command-line interface

benchmarks/                  # reproducible experiment runners
tests/                       # unit and end-to-end tests
docs/                        # architecture and research protocols
schemas/                     # benchmark artifact contract
```

## Research protocol

The full protocol in `docs/end-to-end-experiment-protocol.md` specifies matched trajectories, baselines, disturbances, metrics, required ablations, statistical treatment, reproducibility metadata, and claims boundaries.

A serious study should include ablations for observability, coupling, reliability, redundancy, cost, energy, grouping, adaptive reconfiguration, and resilient estimation.

## Scientific boundaries

This repository supports **controlled simulation research**. It does not establish optimal real-world sensor placement, attack-detection guarantees, safety certification, or deployment-level resilience. The coupling matrices and candidate groups are experimenter-specified model structure. Real deployments require system identification, calibrated noise models, validated sensor reliability, physical experiments, and domain-specific safety analysis.

## Citation

Citation metadata is provided in `CITATION.cff`. Record the exact commit SHA and experiment configuration when reporting results.

## License

MIT
