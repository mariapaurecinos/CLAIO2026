# Overleaf LaTeX Project

## How to compile on Overleaf

1. Zip this entire `paper/` folder: `zip -r paper.zip paper/`
2. Upload to Overleaf as a new project.
3. Overleaf has `llncs.cls` (Springer LNCS) built in.
   If not, download it from Springer and upload alongside `main.tex`.
4. Set compiler to **pdfLaTeX**.
5. Set main document to `main.tex`.

## Structure

```
main.tex                — master file (calls all section files)
abstract.tex
introduction.tex
related_work.tex
methodology.tex
experiments.tex         — experimental design + results (merged for length)
discussion.tex
conclusion.tex
acknowledgements.tex
refs.bib                — bibliography (BibTeX)

tables/
  table_instances.tex   — Table 1: instance summary, three study areas
  table_baselines.tex   — Table 2: fair baseline comparison
  table_allocation.tex  — Table 3: fixed vs. proportional allocation
  table_parameters.tex  — Table 4: calibrated parameters
  table_budget.tex      — Table 5: budget sensitivity
  table_crosscase.tex   — Table 6: cross-context summary
  table_reviewer_response.tex  — Appendix: reviewer-response mapping

figures/
  (25 PNG files copied from experiment outputs)
```

## Source of numerical values

All numbers in the paper were extracted directly from:
- `CLAIO2026/results_publication/{student,city,industrial}/tables/`
- `CLAIO2026/experimentos/{tec,tab,industrial}/outputs/tables/`

No values were estimated or invented.
