# 2. The baseline

The baseline solves the task with three off-the-shelf, **untrained** components —
one per sub-task. Config: [`configs/baseline.yaml`](../configs/baseline.yaml).
Pipeline wiring: `medextract.pipeline.build_pipeline`.

```
note ──► [1] NER ──► spans ──► [2] assertions ──► [3] linking ──► concepts.json
```

## Step 1 — NER with GLiNER (zero-shot)

`medextract/ner/gliner_ner.py`

We need to find and type medical spans without any labelled Vietnamese training
data. **GLiNER** is a zero-shot NER model: you give it natural-language labels and
it tags matching spans. We map descriptive English labels to the task's types
(descriptive labels generalise better than the raw Vietnamese type names):

```yaml
label_map:
  "symptom": "TRIỆU_CHỨNG"
  "disease or diagnosis": "CHẨN_ĐOÁN"
  "medication or drug": "THUỐC"
  "medical test or lab name": "TÊN_XÉT_NGHIỆM"
  "test result or measurement value": "KẾT_QUẢ_XÉT_NGHIỆM"
```

The detection `threshold` (0.35) sets the recall/precision trade-off. Because the
metric punishes both junk and misses, there is a sweet spot around ~15 concepts
per note — low enough to avoid spurious spans, high enough to catch the real ones.
`ner/postprocess.py` then cleans span boundaries (strips leading negation words,
trailing punctuation, section-header words).

## Step 2 — Assertions with ConText (rules)

`medextract/assertions/context_rules.py`

Is a symptom actually present, or negated / about a family member / in the past?
The classic clinical-NLP answer is **ConText**: scan a window around each concept
for trigger cues ("không" → negated, "mẹ/bố" → family, "tiền sử" → historical).
Rules are transparent, need no training, and — on this task — beat an LLM at the
same job.

## Step 3 — Linking with SapBERT + FAISS (retrieval)

`medextract/normalization/retriever.py`

Map each diagnosis to an ICD-10 code and each drug to a RxNorm code. This is a
**retrieval** problem: embed the mention with **SapBERT** (a biomedical sentence
encoder that puts synonyms near each other) and find the nearest concept names in
a **FAISS** index built from the ontology (see `kb/`).

Two small refinements matter:
- **Drugs:** strip route/frequency words ("po", "bid") but keep the strength, and
  filter to strength-specific clinical-drug entries (`filter_tty: [SCD]`) so
  "amlodipine 10 mg" matches "Amlodipine 10 MG Oral Tablet".
- **Low caps:** since candidates are scored by Jaccard, we keep at most 1 drug code
  and 2 diagnosis codes — a spurious extra code halves the score.

The baseline takes the single top retrieved code.

## Run it

```bash
python run.py --config configs/baseline.yaml --input data/dev/input --output out/dev_base
python score.py --pred out/dev_base --gold data/dev -v
```

This is a solid, general starting point (host ≈ 21.8). Its main weakness is step 3:
string-similarity retrieval often puts the correct code in the top few but not at
rank 1. That is exactly what the [improved tier](03_improved.md) fixes.
