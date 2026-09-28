# Results

The subject-level CSV files reported in the paper are placed in this directory:

| File | Content | Produced by |
|---|---|---|
| `eval_subject_level.csv` | fold-averaged accuracy, balanced accuracy and kappa per dataset, mode, decoder, configuration and subject | `scripts/aggregate.py` |
| `eval_statistics.csv` | paired full-minus-reduced tests (non-inferiority, TOST, harm, superiority), unadjusted and Holm-adjusted p-values, decisions and display categories | `scripts/aggregate.py` |
| `eval_operating_points.csv` | fold-level r*, fs*, realised sample counts and input ratios | `scripts/aggregate.py` |
| `eval_training.csv` | best and final epoch, parameter count and fit time of every trained network | `scripts/aggregate.py` |
| `complexity_latency.csv` | parameters, MACs and CPU latency of EEGNet and ATCNet-S per configuration | `scripts/complexity_latency.py` |
| `channel_baselines_classical.csv` | T1 channel subsets compared with same-size subsets ranked by CSP filter weights or drawn at random (classical decoders; supplementary table S4) | earlier evaluation with the same outer splits |
| `estimator_sanity.csv` | synthetic diagnostics of the spatial estimator | `scripts/estimator_sanity.py` |

Re-running the pipeline described in the top-level `README.md` writes the `eval_*` files to `tables/`; `scripts/make_tables_figures.py --results results` regenerates the paper's tables and figures from the files in this directory.
