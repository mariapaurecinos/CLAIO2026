# Final Experiment Report

Generated: 2026-06-29 19:41:49
Notebook:  calibrated_model_selection_and_baselines.ipynb

## Experiment

- **Study area**: industrial — Industrial corridor
- **Grid search outputs**: loaded (81 rows)
- **Baseline outputs**: loaded
- **FORCE_RERUN_GRID**: False
- **FORCE_RERUN_SENSITIVITY**: False
- **FORCE_RERUN_BASELINES**: False

## Selected Calibrated Parameters

- **D_MAX**: 450 m
- **S_MIN**: 200 m
- **beta**: 0.25
- **P_TOTAL**: 42
- **Selection rule**: grid-search balanced scoring

## Performance Summary

- **Proposed coverage rate**: 0.7579
- **Global p-median coverage rate**: 0.8090
- **Coverage retention vs global**: 93.7%
- **Proposed runtime**: 2.5s
- **Global p-median runtime**: 9.4s
- **Runtime reduction vs global**: 73.2%

## Main Conclusion

The calibrated clustered model (spectral + MCA, proportional allocation) achieves 
comparable coverage to the global p-median (93.7% retention) with a 
substantially lower runtime (73.2% reduction), while producing 
operationally interpretable service territories via spectral clustering.
Parameters were selected from sensitivity-supported trade-offs rather than treated 
as optimal values. The fixed-per-cluster allocation of the original paper is replaced
by proportional allocation, which distributes the P=42 opening budget fairly 
according to weighted demand in each cluster.

## Output Files

All tables: `results_calibrated/industrial/tables/`
All figures: `results_calibrated/industrial/figures/`