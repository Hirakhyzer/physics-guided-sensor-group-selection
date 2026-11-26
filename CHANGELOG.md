# Changelog

## v1.0.1 — 2026-11-26

- Validate sensor names, integer state indexes, finite sensing costs, energy requirements and reliability values.
- Reject nonfinite or malformed state-transition and physics-coupling matrices; copy model arrays to prevent changes through caller-owned arrays.
- Validate resource budgets, selection weights, group-count limits and unique group names before ranking sensor combinations.
- Apply the same resource-budget checks to independent-selection baselines, and validate adaptive-reconfiguration reliability/observability inputs.
- Validate estimator variances, measurement dimensions and finite truth/initial-state vectors. NaN retains its documented meaning of a missing measurement, while infinite readings are rejected.
- Validate reproducible scenario controls, disturbance intervals and magnitudes, noise scales and random seeds.
- Add parameterized regression tests for all affected modules and preserve existing synthetic benchmark models.

This maintenance patch strengthens numerical/research-input integrity. It does not introduce new measured hardware energy savings or claim field-validated fault resilience.

## v1.0.0 — 2026-10-12

- Initial versioned physics-guided sensor grouping, constrained selection, synthetic evaluation and reproducing experiments.
