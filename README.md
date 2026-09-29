# Training-derived channel and sampling-rate operating points for motor EEG decoding

Code for the paper *"Training-derived channel and sampling-rate operating points for
motor EEG decoding: a subject-level non-inferiority evaluation"* (J. S. Bindra and
S. Panwar, Indian Institute of Technology Mandi).

For each subject and each training partition, the code estimates an operating point
consisting of a channel count, a channel subset, and a sampling rate, without training
any decoder. The reduced inputs are then evaluated with six decoders (CSP, FBCSP,
Riemannian MDM, tangent-space logistic regression, an EEGNet variant, and a simplified
attention-temporal-convolution network, ATCNet-S) and compared with the full input using
subject-level paired non-inferiority and equivalence tests at a 3-percentage-point margin.

## Installation

Python 3.10 was used for the reported experiments.

```
pip install -r requirements.txt
pip install -e .
```

A CUDA-capable GPU is needed to train the deep networks in practical time.

## Data

All data are public and are downloaded by MOABB on first use; this repository does not
redistribute recordings. Use is subject to the terms of the original providers.

| Dataset (MOABB name) | Reference | Task |
|---|---|---|
| Cho2017 | Cho et al., GigaScience, 2017 | two-class motor imagery |
| Lee2019_MI (OpenBMI) | Lee et al., GigaScience, 2019 | two-class motor imagery, two sessions |
| BNCI2014_001 (BCI Competition IV 2a) | Tangermann et al., Front. Neurosci., 2012 | four-class motor imagery, two sessions |
| Schirrmeister2017 (High-Gamma Dataset) | Schirrmeister et al., Hum. Brain Mapp., 2017 | executed movements and rest |

## Reproducing the analyses

```
export EEGOP_CACHE=~/mne_data/eegop_cache        # optional
python scripts/build_cache.py
python scripts/fetch_channel_info.py cho2017 lee2019_mi schirrmeister2017 bnci2014_001
bash scripts/run_all.sh                           # EEGOP_PARALLEL=1 runs CPU jobs concurrently
python scripts/aggregate.py runs --out tables
python scripts/make_tables_figures.py --results tables --out paper_outputs --runs runs
python scripts/estimator_sanity.py
python scripts/complexity_latency.py 1
```

`scripts/run_evaluation.py --help` lists all options. Each job writes one JSON file per
subject to `runs/<dataset>/<mode>/<tag>/`, including the training/test index hashes,
the operating point, the channels and realised sample counts of every configuration,
and the test-set predictions. Interrupted jobs resume from the last completed subject.

The reported runs used one NVIDIA RTX A5000 (24 GB) and a 32-core CPU. The deep-network
jobs took tens of GPU hours in total; the classical jobs take several CPU hours per dataset.

### Latency measurements

`results/complexity_latency.csv` was produced by `scripts/complexity_latency.py` in a separate
CPU-only environment (Python 3.12.10, PyTorch 2.14.0 CPU build, NumPy 2.5.3; one thread on an
Intel Core i7-14700K), listed in `requirements-latency.txt`. The training and evaluation
environment is the one in `requirements.txt`. Each network is timed with batch size 1 over
300 forward passes after 50 warm-up passes (5 warm-up passes for networks with at least
10^9 MACs); the first temporal convolution is evaluated by FFT. Parameter and MAC counts do
not depend on the environment; latencies depend on the hardware and library versions.

## Reported results

The subject-level results reported in the paper are in `results/` (see
`results/README.md`). The tables and figures of the paper can be regenerated from these
files without re-running the evaluation:

```
python scripts/make_tables_figures.py --results results --out paper_outputs
```

## Operating point

Computed from the training trials of one split only (`src/eegop/estimators.py`):

- **Spatial score.** 8-30 Hz band-pass, class-mean trial covariances, whitening by the
  loaded mixture covariance (loading `1e-4 * trace / C`), and a per-direction score
  `m_j = -1/2 sum_k pi_k log2(w_j' Sigma_k w_j)` (`score="standard"`). The score equals
  a contrast term (the Jensen gap of the projected class variances) plus a loading term
  `1/2 log2(1 + eps/lambda_j)`; `score="jensen"` computes the contrast term alone.
- **Channel count.** Participation ratio of the positive scores for two-class data;
  smallest number of directions reaching 95% of the total score for four classes.
- **Channel subset.** Channels with the largest summed squared whitening weights over
  the selected directions.
- **Sampling rate.** `fs* = min(max(2 (B + 2/T), k C / T), fs)` with `k = 7`, where `B` is
  an estimated task band edge, `T` the epoch length, and `C` the channel count.

T1 applies the channel subset at the full rate, T2 all channels at `fs*`, and T3 both.

## Evaluation protocol

- Within-session: stratified five-fold cross-validation per subject (seed 42). The
  operating point is estimated on the training folds and applied unchanged to the test fold.
- Cross-session: estimation and training on one session, testing on the other.
- Deep networks: 80/20 split of the training data for early stopping on validation
  cross-entropy (batch 32, at most 100 epochs, patience 20); the test data are scored once.
- Configurations: `full`, `T1`, `T2`, `T3`, rate factor `k = 5` and `k = 10` (`T2_k5`, `T2_k10`),
  fixed 128 Hz (`F128`), a fixed 10-10 sensorimotor montage (`SM`), the contrast-only
  channel score (`T1_JG`), and T3 with 0.75 and 1.25 times the estimated channel count.

## Statistics

For each subject, `d = acc_full - acc_reduced` is averaged over folds. Non-inferiority
(primary) is a one-sided paired t-test of `E[d] < 0.03`; equivalence uses two one-sided
tests with margin 0.03. Holm adjustment is applied within prespecified families (see
`scripts/aggregate.py`). Intervals are two-sided 90% t intervals.

## Repository layout

```
src/eegop/estimators.py    operating-point estimators
src/eegop/models.py        CSP, FBCSP, MDM, EEGNet variant, ATCNet-S
src/eegop/protocol.py      decoder training/evaluation, tangent-space decoder
src/eegop/stats.py         paired non-inferiority / equivalence tests, Holm adjustment
src/eegop/lee_sessions.py  session-preserving Lee2019_MI loader for MOABB 1.5
scripts/build_cache.py     epoch the public datasets through MOABB
scripts/fetch_channel_info.py  channel names (needed for the sensorimotor montage)
scripts/run_evaluation.py  one evaluation job (dataset x mode x decoders x configurations)
scripts/run_all.sh         all jobs reported in the paper
scripts/aggregate.py       subject-level tables and statistics
scripts/make_tables_figures.py  tables, figures and numbers reported in the paper
scripts/estimator_sanity.py, scripts/real_data_estimator_diag.py  estimator diagnostics
scripts/complexity_latency.py  parameters, MACs and CPU latency
tests/                     unit tests (statistics, estimators, FFT convolution)
results/                   subject-level result tables reported in the paper
```

## Tests

```
pytest -q
```

## Citation

Citation information will be added upon publication; `CITATION.cff` describes the software.

## License

MIT (see `LICENSE`).
