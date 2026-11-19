# Functional Tests 8.1 – 13.9

FinTech 545 — Quantitative Risk Management

Python implementation of functional tests 8.1 through 13.9. Tests 1.1–7.6 are
in the sibling folder `../Test 1.1-7.6/`.

## Contents

| path | what it is |
|---|---|
| `run_tests.py` | Runs tests 8.1–13.9, writes results to `output/`, compares to expected files. |
| `risklib/` | Library modules used by these tests (includes earlier helpers they depend on). |
| `data/` | Inputs and expected outputs for tests 8–13. |
| `output/` | Our results. Regenerated on every run. |

### `risklib` modules used here

| module | tests |
|---|---|
| `risk_stats.py` | 8.1–8.6, 9.1, 13.9 |
| `optimizers.py` | 10.1–10.4 |
| `attribution.py` | 11.1, 11.2 |
| `options.py` | 12.1–12.3 |
| `multivariate_t.py` | 13.1–13.4 |
| `copula.py` | 9.1, 13.5–13.9 |
| `fitting.py`, `simulation.py`, `psd.py` | dependencies of the above |

## Requirements

```bash
pip install numpy pandas scipy
```

## Running

```bash
python run_tests.py                 # all 25 tests, ~50 seconds (12.3 dominates)
python run_tests.py 8.1 13.7        # a subset
```

## Tolerances

| tests | tolerance |
|---|---|
| 8.1, 8.4, 11.x, 12.1, 12.3, 13.1–13.3, 13.8 | 1e-8 relative |
| 8.2, 8.5, 13.4–13.7 | 1e-4 relative |
| 10.x, 12.2 | 5e-3 relative |
| 8.3, 8.6 | analytical cross-check vs 8.2 / 8.5 (within 10%) |
| 9.1, 13.9 | 15% relative (RNG differs from Julia) |

## Notes

- **8.3 / 8.6** are graded against analytical 8.2 / 8.5 per Tests.xlsx.
- **10.3 / 10.4** use `test5_3.csv` (non-PSD), matching `test_setup.jl`.
- **12.2 theta** is `∂V/∂T` from FD on maturity, as in the expected file.
- **12.3** discrete-dividend American tree is ~23 s per option in pure Python.
