# REASONIX.md — Project Two

## Stack
- **Python 3.11+** — no package manager at root (no pyproject.toml / setup.py / requirements.txt)
- **PyTorch** — GPU evaluation via `torch` (runners check `torch.cuda.is_available()`)
- **pandas + numpy** — all data processing and metric computation
- **matplotlib** — report figures via `scripts/generate_report_figures.py`
- **Streamlit** — SimTrading dashboard only (`SimTrading/` subdir, isolated app)

## Layout
- `configs/experiments.yaml` — **JSON** despite `.yaml` extension; paths, datasets, execution plan
- `data/` — one subdir per dataset, each with `processed/metadata.json` + feature CSVs
- `results/` — `{dataset}/{Model}/{setting}[_{mode}]/` → `metrics.csv`, `predictions.parquet`
- `runners/` — eval entrypoints + `aggregate_results.py`; `runners/builders/` — dataset prep
- `scripts/` — 5-stage pipeline (`stage_1_download` … `stage_5_convert_simtrading`) + `run_all.py` orchestrator
- `SimTrading/` — standalone Streamlit backtest dashboard, own `requirements.txt`

## Commands
```bash
# Full pipeline
python scripts/run_all.py

# Single stage, filtered
python scripts/run_all.py --stage 3 --datasets electricity --models DLinear

# SimTrading (from SimTrading/)
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests -v
```

## Conventions
- **Every `.py` file** starts with `from __future__ import annotations` and uses `from pathlib import Path`
- **Settings** encoded as `H{history}_F{forecast}` strings (e.g. `H60_F1`, `H336_F96`)
- **Mode** is a separate field (`zero_shot`, `training`, `text_only`) for ECHO models only
- **`FINANCIAL_DATASETS`** = `{"fnspid", "oiletf", "oiletf_intraday"}` — used for special-cased metric logging
- **JSON everywhere** for config/metadata/output; `write_json` uses atomic `.tmp` → replace
- **`sys.path` bootstrap** in scripts — `scripts/` and `runners/` inserted into `sys.path` so `from common import …` works

## Watch out for
- **`configs/experiments.yaml` is JSON**, not YAML. The `.yaml` extension is misleading — parse with `json.load()`.
- **Separate conda envs per runner** — `chronos`, `aurora`, `tslib`, `oiletf` each need their own Python (configured in `configs/experiments.yaml`)
- **HF mirror** — `chronos_runner.py` and `chronos_echo_runner.py` set `HF_ENDPOINT=https://hf-mirror.com` before importing HF libs
- **No tests at root** — only `SimTrading/tests/` has unit tests; runner/script correctness is validated by checking output files exist
- **`data/` and `results/` in `.gitignore`** — git-tracked project does not include datasets or evaluation outputs
