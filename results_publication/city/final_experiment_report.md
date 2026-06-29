# Final Experiment Report — Less-Developed City (Villahermosa, Tabasco)

Generated: 2026-06-25 16:39:01
Experiment: city

## Instance

| Parameter | Value |
|-----------|-------|
| Demand nodes | 2,074 |
| Candidate blocks | 1,256 |
| Existing stores | 18 |
| Clusters | 6 |
| P_total (budget) | 42 |
| D_MAX (m) | 366 |
| S_MIN (m) | 240 |
| beta | 0.25 |

## Calibrated Parameters

| Parameter | Original | Calibrated |
|-----------|----------|------------|
| D_MAX | 366 | 366.0 |
| S_MIN | 240 | 200.0 |
| beta | 0.25 | 0.25 |

## Results Summary

| Model | Coverage (%) | W-Coverage (%) | W-Mean Dist (m) | Runtime (s) |
|-------|-------------|----------------|-----------------|-------------|
| Proposed model | 59.26 | 64.47 | 216.3 | 11.2 |
| Global P-Median (B1) | 60.51 | N/A | 216.7 | 11.4 |

## Key Performance Metrics

- **Coverage retention vs global**: 97.9%
- **Runtime reduction vs global**: 1.5%
- **Weighted distance difference vs global**: -0.4 m

## Main Conclusion

The proposed clustered model retains 97.9% of the global p-median coverage
while reducing runtime by 1.5%. The weighted mean assignment distance differs
by only -0.4 m from the global optimum.

Parameters were calibrated through multi-objective sensitivity analysis (Pareto frontier
over 81 grid-search combinations) rather than fixed a priori.
