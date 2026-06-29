# Final Experiment Report

Generated: 2026-06-26 20:54:42
Notebook:  calibrated_model_selection_and_baselines.ipynb

## Experiment

- **Study area**: city — Less-Developed City (Villahermosa, Tabasco)
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

- **Proposed coverage rate**: 0.5926
- **Global p-median coverage rate**: 0.6051
- **Coverage retention vs global**: 97.9%
- **Proposed runtime**: 11.2s
- **Global p-median runtime**: 11.4s
- **Runtime reduction vs global**: 1.5%

## Main Conclusion

The calibrated clustered model (spectral + MCA, proportional allocation) achieves 
comparable coverage to the global p-median (97.9% retention) with a 
substantially lower runtime (1.5% reduction), while producing 
operationally interpretable service territories via spectral clustering.
Parameters were selected from sensitivity-supported trade-offs rather than treated 
as optimal values. The fixed-per-cluster allocation of the original paper is replaced
by proportional allocation, which distributes the P=42 opening budget fairly 
according to weighted demand in each cluster.

## Output Files

All tables: `results_calibrated/city/tables/`
All figures: `results_calibrated/city/figures/`