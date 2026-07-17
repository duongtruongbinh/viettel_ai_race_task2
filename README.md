# Medical Concept Extraction from Vietnamese Clinical Notes

A teaching repo for **Task 2 of the Viettel AI Race 2026**: read a free-form
Vietnamese clinical note and produce structured medical concepts — each with a
type, clinical assertions, and a link to a standard code (ICD-10 for diagnoses,
RxNorm for drugs).

It is built to be learned from, top to bottom: a general **baseline** you can run
on a free Colab GPU, then one focused **improvement**. No training required.

---

## The problem

Given a note like:

```
Bệnh nhân nam 60 tuổi, tiền sử đái tháo đường type 2. Không sốt, không ho.
Đang dùng metformin 500 mg po bid. Mẹ bị ung thư vú.
```

extract every medical concept and, for each, produce:

| field | example |
|---|---|
| **text** + **position** | `"đái tháo đường type 2"` at `[38, 59]` |
| **type** (one of 5) | `CHẨN_ĐOÁN` (diagnosis) |
| **assertions** | `isHistorical` (past), `isNegated` (không sốt), `isFamily` (mẹ) |
| **candidates** (codes) | diagnosis → ICD-10 `E11.9`; drug → RxNorm `861007` |

The five types: `TRIỆU_CHỨNG` (symptom), `TÊN_XÉT_NGHIỆM` (test name),
`KẾT_QUẢ_XÉT_NGHIỆM` (test result), `CHẨN_ĐOÁN` (diagnosis), `THUỐC` (drug).

**Scoring.** `final = 0.3·(1 − WER) + 0.3·J_assertion + 0.4·J_candidates`, ×100.
A concept whose text or type is wrong scores 0 on all three parts — so predicting
junk hurts as much as missing a concept. Full details in
[`docs/01_problem.md`](docs/01_problem.md).

---

## The approach: three classic NLP steps

The task decomposes into the three pillars of clinical information extraction, and
we solve each with a standard, general tool:

| Step | What it does | Tool |
|---|---|---|
| **1. NER** | find + type the concepts | **GLiNER** zero-shot (no training) |
| **2. Assertion** | negated? family? historical? | **ConText** rule algorithm |
| **3. Linking** | concept → ICD-10 / RxNorm code | **SapBERT + FAISS** retrieval |

### Baseline → Improved

| tier | recipe | dev / host |
|---|---|---|
| **Baseline** ([`docs/02_baseline.md`](docs/02_baseline.md)) | steps 1–3 above; retrieval picks the top-1 code | host ≈ 21.8 |
| **Improved** ([`docs/03_improved.md`](docs/03_improved.md)) | add a local LLM that **reranks** the retrieved codes (retrieve-then-rerank) | host ≈ 24.5–26 |

The improvement targets the hardest step: retrieval gets the correct code into the
top-k but not always at rank 1, so a small LLM reads the note and picks the right
one — constrained to the retrieved codes, so it can never invent an invalid code.
The 8B model loads **4-bit** (~6 GB) to fit a free Colab T4.

Domain-specific tricks that squeezed out the last point or two live in
[`docs/appendix_host_tuning.md`](docs/appendix_host_tuning.md) — kept separate
because they are competition tuning, not general method.

---

## Quickstart

```bash
# 1. environment  (see INSTALL.md for details, incl. the knowledge bases)
conda create -n medextract python=3.10 -y && conda activate medextract
pip install -r requirements.txt && pip install -e .

# 2. build the knowledge bases (one-time; sources are license-gated, see INSTALL.md)
python -m medextract.kb.build_rxnorm
python -m medextract.kb.build_icd
python -m medextract.kb.index --device auto

# 3. run the baseline on the bundled sample notes
python run.py --config configs/baseline.yaml --input data/sample_input --output out/demo

# 4. score on the dev set
python run.py --config configs/baseline.yaml --input data/dev/input --output out/dev_base
python score.py --pred out/dev_base --gold data/dev

# 5. improved pipeline + a submission zip
python run.py --config configs/improved.yaml --input <your test dir> --output out/sub --zip
```

Prefer a notebook? [`notebooks/colab_baseline.ipynb`](notebooks/colab_baseline.ipynb)
and [`notebooks/colab_improved.ipynb`](notebooks/colab_improved.ipynb) run the whole
thing on a free Colab T4.

## Layout

```
run.py            # run a config over an input dir -> per-file JSON (+ optional zip)
score.py          # score predictions against dev gold
serve.py          # optional: POST /extract API (same pipeline)
configs/          # baseline.yaml, improved.yaml
src/medextract/   # the library: ner, assertions, normalization, kb, llm, scoring
prompts/          # LLM rerank + assertion prompt templates
data/dev/         # 20-note dev set + gold;  data/sample_input/ = 3 demo notes
notebooks/        # Colab notebooks
docs/             # 01_problem, 02_baseline, 03_improved, appendix_host_tuning
tests/            # unit tests
```

## Constraints (from the competition)

Fully self-hosted and offline: no external LLM APIs, models ≤ 9B parameters. The
default LLM is Qwen3-8B (8.2B). Everything here honours that.
