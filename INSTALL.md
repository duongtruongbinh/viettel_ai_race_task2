# Installation

## 1. Environment

```bash
conda create -n medextract python=3.10 -y
conda activate medextract
pip install -r requirements.txt
pip install -e .
```

Quick check:

```bash
python -c "import medextract, gliner, faiss, transformers; print('env ok')"
pytest -q
```

## 2. Knowledge bases (one-time)

The linking step matches mentions against two ontologies: **RxNorm** (drugs) and
**ICD-10** (diagnoses). These source files are license-gated and are **not**
shipped with the repo — download them yourself and drop them in `data/kb/raw/`.

### RxNorm

Get `RXNCONSO.RRF` from the NLM RxNorm release (UMLS/UTS account, free) and place
it at `data/kb/raw/RXNCONSO.RRF`. A Kaggle CSV export
(`data/kb/raw/rxnorm_rxnconso.csv`, same RXNCONSO columns) also works.

```bash
python -m medextract.kb.build_rxnorm      # -> data/kb/processed/rxnorm_terms.parquet
```

### ICD-10

Either source works:
- **Vietnamese QĐ 4469/QĐ-BYT** Excel → any `*.xlsx`/`*.xls` in `data/kb/raw/`
  (columns Mã / Mô tả tiếng Việt / Mô tả tiếng Anh). Preferred — its Vietnamese
  names match Vietnamese diagnosis mentions best.
- **WHO ICD-10 English** list — downloaded and cached automatically if no Excel is
  present (multilingual SapBERT still links Vietnamese mentions to English names).

```bash
python -m medextract.kb.build_icd         # -> data/kb/processed/icd_terms.parquet
```

### Build the FAISS index

Embeds every term with SapBERT and builds the nearest-neighbor indexes:

```bash
python -m medextract.kb.index --device auto   # -> data/kb/processed/{RXNORM,ICD10}.faiss
```

`--device auto` picks a free GPU (falls back to CPU; slower but works).

## 3. Run

```bash
python run.py --config configs/baseline.yaml --input data/sample_input --output out/demo
```

See the [README](README.md) for the full workflow, and `docs/` for the walkthrough.

## Notes

- **GPU:** the baseline runs on a free Colab T4 (~3 GB) or CPU. The improved tier
  loads an 8B LLM; with `load_in_4bit: true` (the default in `configs/improved.yaml`)
  it needs ~6 GB and fits a T4. For a full-precision run set `load_in_4bit: false`
  and `device: wait` (needs ~18 GB).
- **Models:** GLiNER, SapBERT, and Qwen3-8B download from the Hugging Face Hub on
  first use and are cached. To stay fully offline, pre-download them or point
  `llm.model` at a local path.
