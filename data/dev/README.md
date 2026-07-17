# Dev set — Task 2: Ontological Reasoning in Medical Knowledge Retrieval

A small, fully-annotated development set (20 notes) for smoke-testing your
pipeline and your metric before you touch the private test set. Inputs are
free-form Vietnamese clinical notes; gold labels follow the exact submission JSON
schema.

## Contents

```
dev/
├── input/   20 clinical notes:      1.txt .. 20.txt
├── gold/    20 gold annotations:    1.json .. 20.json  (submission schema)
└── README.md
```

20 notes · 160 gold concepts:

| type | count |
|---|---|
| TRIỆU_CHỨNG (symptom) | 72 |
| CHẨN_ĐOÁN (diagnosis, +ICD-10) | 27 |
| THUỐC (drug, +RxNorm) | 26 |
| TÊN_XÉT_NGHIỆM (test name) | 20 |
| KẾT_QUẢ_XÉT_NGHIỆM (test result) | 15 |

Assertion coverage: `isHistorical` ×27, `isNegated` ×23, `isFamily` ×5.

The set deliberately exercises each metric: negation-heavy notes, family history,
pre-admission drug/history lists, pure lab panels (empty assertions/candidates),
a repeated span, messy input (double spaces, abbreviations), and multi-code
diagnoses.

Every `position` is a 0-indexed, end-exclusive character offset with
`input[start:end] == text` exactly.

## How to use

```bash
# sanity: scoring gold against itself is a perfect 100.00
python score.py --pred data/dev/gold --gold data/dev

# run your pipeline, then score its output
python run.py --config configs/baseline.yaml --input data/dev/input --output out/dev_base
python score.py --pred out/dev_base --gold data/dev -v      # -v = per-record rows
```

The scoring formula (see `docs/01_problem.md`):
`final = 0.3·text + 0.3·assertions + 0.4·candidates`, ×100.

This dev set is for local iteration, not a leaderboard proxy — 20 hand-built
notes won't match the private-test distribution. Use it to validate your output
schema and offsets, debug your scorer, and catch regressions.
