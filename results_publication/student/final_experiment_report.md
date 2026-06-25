# Final Experiment Report — Tecnológico de Monterrey student area

Generated: 2026-06-24 20:02:54
Experiment: student

## Instance

| Parameter | Value |
|-----------|-------|
| Demand nodes | 4,731 |
| Candidate blocks | 3,031 |
| Existing stores | 67 |
| Clusters | 6 |
| P_total (budget) | 42 |
| D_MAX (m) | 366 |
| S_MIN (m) | 240 |
| beta | 0.25 |

## Calibrated Parameters

| Parameter | Original | Calibrated |
|-----------|----------|------------|
| D_MAX | 366 | 300.0 |
| S_MIN | 240 | 200.0 |
| beta | 0.25 | 0.1 |

## Results Summary

| Model | Coverage (%) | W-Coverage (%) | W-Mean Dist (m) | Runtime (s) |
|-------|-------------|----------------|-----------------|-------------|
| Original proposed | 45.57 | 47.79 | 223.3 | N/A |
| Revised proposed | 47.77 | 49.38 | 227.8 | 25.8 |
| Global P-Median (B1) | 47.83 | N/A | 229.0 | 70.6 |

## Key Performance Metrics

- **Coverage retention vs global**: 99.9%
- **Runtime reduction vs global**: 63.4%
- **Weighted distance difference vs global**: -1.2 m

## Main Conclusion

The revised clustered model retains 99.9% of the global p-median coverage
while reducing runtime by 63.4%. The proportional allocation corrects the
equity problem of fixed-7 assignment. The weighted mean assignment distance differs
by only -1.2 m from the global optimum.

Parameters were calibrated through multi-objective sensitivity analysis (Pareto frontier
over 81 grid-search combinations) rather than fixed a priori.
