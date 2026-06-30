# Final Experiment Report

Generated: 2026-06-29 19:36:29
Notebook:  calibrated_model_selection_and_baselines.ipynb

## Experiment

- **Study area**: student — Tecnológico de Monterrey student area
- **Grid search outputs**: loaded (81 rows)
- **Baseline outputs**: loaded
- **FORCE_RERUN_GRID**: False
- **FORCE_RERUN_SENSITIVITY**: False
- **FORCE_RERUN_BASELINES**: False

## Selected Calibrated Parameters

- **D_MAX**: 300 m
- **S_MIN**: 200 m
- **beta**: 0.1
- **P_TOTAL**: 42
- **Selection rule**: grid-search balanced scoring

## Performance Summary

- **Proposed coverage rate**: 0.4777
- **Global p-median coverage rate**: 0.4783
- **Coverage retention vs global**: 99.9%
- **Proposed runtime**: 27.5s
- **Global p-median runtime**: 76.3s
- **Runtime reduction vs global**: 63.9%

## Main Conclusion

The calibrated clustered model (spectral + MCA, proportional allocation) achieves 
comparable coverage to the global p-median (99.9% retention) with a 
substantially lower runtime (63.9% reduction), while producing 
operationally interpretable service territories via spectral clustering.
Parameters were selected from sensitivity-supported trade-offs rather than treated 
as optimal values. The fixed-per-cluster allocation of the original paper is replaced
by proportional allocation, which distributes the P=42 opening budget fairly 
according to weighted demand in each cluster.

## Output Files

All tables: `results_calibrated/student/tables/`
All figures: `results_calibrated/student/figures/`