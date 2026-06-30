# Final Experiment Report — Industrial study area

Generated: 2026-06-29 15:11:36
Experiment: industrial

## Instance

| Parameter | Value |
|-----------|-------|
| Demand nodes | 1,450 |
| Candidate blocks | 969 |
| Existing stores | 93 |
| Clusters | 11 |
| P_total (budget) | 42 |
| D_MAX (m) | 366 |
| S_MIN (m) | 240 |
| beta | 0.25 |

## Calibrated Parameters

| Parameter | Original | Calibrated |
|-----------|----------|------------|
| D_MAX | 366 | 450.0 |
| S_MIN | 240 | 200.0 |
| beta | 0.25 | 0.1 |

## Results Summary

| Model | Coverage (%) | W-Coverage (%) | W-Mean Dist (m) | Runtime (s) |
|-------|-------------|----------------|-----------------|-------------|
| Proposed model | 75.79 | 82.51 | 194.4 | 2.5 |
| Global P-Median (B1) | 80.90 | N/A | 200.4 | 9.4 |

## Key Performance Metrics

- **Coverage retention vs global**: 93.7%
- **Runtime reduction vs global**: 73.2%
- **Weighted distance difference vs global**: -6.0 m

## Main Conclusion

The proposed clustered model retains 93.7% of the global p-median coverage
while reducing runtime by 73.2%. The weighted mean assignment distance differs
by only -6.0 m from the global optimum.

Parameters were calibrated through multi-objective sensitivity analysis (Pareto frontier
over 81 grid-search combinations) rather than fixed a priori.
