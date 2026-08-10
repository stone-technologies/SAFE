# SAFE-ALPHA replication package

This package supports "SAFE-ALPHA: Anytime-Valid Certification for Adaptive Strategy Search."

## Contents

- `code/`: causal strategy execution, proposal policy, e-process gate,
  baselines, simulations, and figure generation.
- `data/raw/`: the archived Kenneth R. French Data Library ZIP files used in
  the paper
- `data/processed/`: parsed daily panels.
- `results/`: proposal and certification ledgers and all reported estimates.
- `paper/`: anonymous ICAIF source, the extended proof manuscript, figures,
  and generated tables.
- `tests/`: 22 tests for inference, execution, portfolio accounting, matched
  terminal containment, factor residualization.

## Running



```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
PYTHONPATH=code MPLCONFIGDIR=/tmp/safe-alpha-mpl \
  python code/run_experiments.py
PYTHONPATH=code MPLCONFIGDIR=/tmp/safe-alpha-mpl \
  python code/make_figures.py
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 \
  python code/run_empirical_finish.py --jobs 8
PYTHONPATH=code MPLCONFIGDIR=/tmp/safe-alpha-mpl \
  python code/empirical_diagnostics.py
PYTHONPATH=code MPLCONFIGDIR=/tmp/safe-alpha-mpl \
  python code/make_empirical_finish_figures.py
```


```bash
PYTHONPATH=code python code/run_power_upgrade.py \
  --output results/power_upgrade_development_v4.csv \
  --seed-start 41000000 --repetitions 100 --jobs 4
PYTHONPATH=code python code/run_power_upgrade_validation.py
PYTHONPATH=code python code/run_power_upgrade_v2_holdout.py
PYTHONPATH=code python code/run_conservative_baseline.py \
  --design campaign75-geometric-daily --jobs 4
PYTHONPATH=code python code/run_power_upgrade_final_heavy_null.py
python code/build_final_manifest.py
```

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error extended.tex
```