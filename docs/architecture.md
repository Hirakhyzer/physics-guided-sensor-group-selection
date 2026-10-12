# Architecture

```text
             Physical model A + coupling graph
                         |
          +--------------+--------------+
          |                             |
    Sensor metadata                Candidate groups
 cost / energy / reliability       physical semantics
          |                             |
          +--------------+--------------+
                         |
                Physics scoring
       observability / coupling / redundancy
                         |
              Marginal group selector
                 cost + energy budgets
                         |
                Selected sensors
                         |
       +-----------------+------------------+
       |                                    |
 Standard Kalman estimator        Innovation-gated estimator
       |                                    |
       +-----------------+------------------+
                         |
              State-estimation RMSE
                         |
             Fault / attack scenarios
                         |
             Reliability degradation
                         |
               Adaptive re-selection
                         |
                 Paired evaluation
```

## Design decisions

### Groups are semantic candidates, sensors are physical resources

Candidate groups encode domain relationships. Resource accounting is performed on the union of physical sensors, so overlapping groups do not pay twice for the same sensor.

### Selection uses marginal utility

A group's value is recomputed after each selection against the current sensor union. This prevents repeated observability or coupling credit when groups overlap.

### Internal utility is not the final metric

Selection utility is useful for optimization, but experiments judge a sensor set by downstream state-estimation error, robustness, observability, and resource use.

### Fault simulation is separate from robust estimation

Dropout, bias, drift and freeze scenarios modify telemetry. The resilient estimator may reject inconsistent measurements, but rejection is intentionally not labeled as attack detection.

### Adaptation fails visibly

After sensors become unavailable or unreliable, reconfiguration returns the achieved observability fraction and an explicit boolean indicating whether the requested target remains satisfied. The framework does not silently claim resilience when the surviving instrumentation cannot meet the target.
