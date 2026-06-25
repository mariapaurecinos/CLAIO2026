# Final Experiment Report — Less-Developed City (Villahermosa, Tabasco)

Generated: 2026-06-24 20:20:27
Experiment: city

## Instance

| Parameter | Value |
|-----------|-------|
| Demand nodes | 2,074 |
| Candidate blocks | 1,256 |
| Existing stores | 18 |
| Clusters | 5 |
| P_total (budget) | 42 |
| D_MAX (m) | 366 |
| S_MIN (m) | 240 |
| beta | 0.25 |

## Calibrated Parameters

| Parameter | Original | Calibrated |
|-----------|----------|------------|
| D_MAX | 366 | 450.0 |
| S_MIN | 240 | 180.0 |
| beta | 0.25 | 0.25 |

## Results Summary

| Model | Coverage (%) | W-Coverage (%) | W-Mean Dist (m) | Runtime (s) |
|-------|-------------|----------------|-----------------|-------------|
| Original proposed | 54.77 | 59.43 | 217.6 | N/A |
| Revised proposed | 59.55 | 64.84 | 215.6 | 11.0 |
| Global P-Median (B1) | 60.51 | N/A | 216.7 | 11.3 |

## Key Performance Metrics

- **Coverage retention vs global**: 98.4%
- **Runtime reduction vs global**: 2.8%
- **Weighted distance difference vs global**: -1.2 m

## Main Conclusion

The revised clustered model retains 98.4% of the global p-median coverage
while reducing runtime by 2.8%. The proportional allocation corrects the
equity problem of fixed-7 assignment. The weighted mean assignment distance differs
by only -1.2 m from the global optimum.

Parameters were calibrated through multi-objective sensitivity analysis (Pareto frontier
over 48 grid-search combinations) rather than fixed a priori.
