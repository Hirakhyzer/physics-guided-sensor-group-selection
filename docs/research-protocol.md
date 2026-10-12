# Research Protocol

## Objective

Evaluate whether physics-guided sensor groups produce better state observability and sensor-dropout resilience than selections driven only by sensor cost or individual sensor scores.

## Experimental factors

- sensing cost budget;
- sensing energy budget;
- maximum selected group count;
- group construction strategy;
- reliability assumptions;
- number of failed sensors;
- system coupling strength.

## Primary outcomes

1. selected sensor count;
2. total cost and energy;
3. observability rank and fraction;
4. worst-case observability under sensor dropout;
5. mean observability under dropout;
6. group utility under fixed weights.

## Baselines

Future experiments should compare at least:

- cheapest-sensor selection;
- reliability-only ranking;
- individual observability ranking;
- random sensor subsets under matched budget;
- physics-guided grouped selection.

Budget matching is essential. A method must not be credited for better observability simply because it uses more sensors or more energy.

## Ablations

The multi-objective selector should be evaluated with each term removed in turn:

- no physics coupling;
- no observability term;
- no reliability term;
- no redundancy term;
- no sensing-cost penalty;
- no energy penalty.

## Robustness

For each selected set, enumerate single-sensor dropout when tractable and report worst-case and mean retained observability. Multi-sensor dropout should be added for larger experiments with explicit combinatorial limits or sampled failure sets.

## Validity boundaries

The current benchmark is a synthetic linear reduced-order model. Observability is evaluated from the represented state-transition and measurement matrices. Physical coupling scores encode model structure supplied by the experimenter and are not independently validated physical laws. Results therefore demonstrate algorithmic behavior under controlled assumptions, not deployment-level sensing guarantees.
