# End-to-End Experiment Protocol

## Objective

Evaluate whether physics-guided **group** selection produces sensor sets that preserve state-estimation quality and robustness under resource constraints better than non-grouped baselines.

The primary outcome is not the selector's internal utility. It is downstream estimation performance on matched simulated trajectories.

## Benchmark domains

The repository contains three reduced-order linear research models:

1. coupled hydraulic-style process;
2. thermal building/HVAC process;
3. battery electro-thermal process.

Each domain contains five sensors, explicit resource/reliability metadata, a state-transition matrix, a coupling matrix, and physically motivated candidate groups. These models are intentionally small enough for transparent experiments. They are not validated operational twins.

## Selection methods

Compare at least:

- physics-guided marginal group selection;
- observability-greedy independent selection;
- reliability-first selection;
- cheapest-first selection;
- membership-frequency selection where applicable.

All methods receive the same cost and energy budgets.

## Matched scenarios

For every domain and random seed, evaluate the same generated truth trajectory under:

- clean sensing;
- sensor dropout;
- additive bias;
- gradual drift;
- frozen measurement.

Within an evaluation case, the full sensor trace is generated first and each method receives a subset of its columns. This avoids accidentally comparing methods on different physical trajectories.

## Estimators

Two downstream estimators are reported:

1. standard linear Kalman estimator with missing-value support;
2. innovation-gated resilient estimator that rejects measurements inconsistent with predicted uncertainty.

The innovation gate is a defensive research mechanism. Rejection is not equivalent to attack attribution.

## Metrics

Report:

- selected sensor count;
- total cost and sensing energy;
- observability fraction;
- clean RMSE;
- scenario RMSE;
- mean and worst-case RMSE;
- resilient-estimator RMSE;
- rejected measurement count;
- dropout observability degradation;
- adaptive reconfiguration observability;
- paired candidate-minus-baseline RMSE deltas.

## Statistical comparison

Use matched domain/scenario/seed observations. `paired_bootstrap_delta` resamples paired deltas and reports a descriptive 95% percentile interval and candidate win rate. Do not treat repeated time steps from one trajectory as independent experimental replicates.

## Required ablations

A serious evaluation should remove one component at a time:

- no coupling term;
- no observability term;
- no reliability term;
- no redundancy term;
- no cost penalty;
- no energy penalty;
- individual sensor selection instead of groups;
- static selection instead of adaptive reconfiguration;
- standard estimator instead of innovation gating.

## Reproducibility

For every reported table or figure retain:

- repository commit SHA;
- Python version;
- domain;
- random seeds;
- budgets;
- selector weights;
- process and measurement noise;
- disturbance type, magnitude and interval;
- estimator configuration;
- generated JSON artifact.

## Claims boundary

The project can support claims about controlled synthetic experiments. It cannot establish real-world sensor placement optimality, attack detection guarantees, deployment safety, or transfer to a physical installation without external validation.
