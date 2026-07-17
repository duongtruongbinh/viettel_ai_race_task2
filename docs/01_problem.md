# 1. The problem

## Input and output

The input is one free-form Vietnamese clinical note as plain text. The output is a
JSON list of medical concepts. Each concept is:

```json
{
  "text": "đái tháo đường type 2",
  "position": [38, 59],
  "type": "CHẨN_ĐOÁN",
  "assertions": ["isHistorical"],
  "candidates": ["E11.9"]
}
```

- **text / position** — the exact surface string and its character offsets
  (0-indexed, end-exclusive: `note[start:end] == text`).
- **type** — one of five:
  - `TRIỆU_CHỨNG` — symptom
  - `TÊN_XÉT_NGHIỆM` — test / lab name
  - `KẾT_QUẢ_XÉT_NGHIỆM` — test result / value
  - `CHẨN_ĐOÁN` — diagnosis
  - `THUỐC` — drug
- **assertions** — clinical context flags, only for symptoms/diagnoses/drugs:
  - `isNegated` — stated as absent ("không sốt")
  - `isFamily` — about a relative ("mẹ bị ung thư vú")
  - `isHistorical` — in the past / pre-admission ("tiền sử...")
- **candidates** — standard codes: ICD-10 for `CHẨN_ĐOÁN`, RxNorm for `THUỐC`.
  Symptoms/tests/results have no codes.

## Scoring

```
final = 0.3·(1 − WER) + 0.3·J_assertion + 0.4·J_candidates      (×100)
```

- **WER** — word error rate between predicted and gold concept text.
- **J_assertion** — Jaccard overlap of the assertion sets.
- **J_candidates** — Jaccard overlap of the code sets (weighted by how many codes
  a gold concept has).

Two things follow directly from the formula and drive every design choice:

1. **A wrong concept is doubly penalised.** A prediction whose text or type does
   not match any gold concept can't match its twin, so it scores 0 on all three
   parts *and* inflates the denominator. Emitting junk is as costly as missing a
   real concept — precision matters as much as recall.
2. **Candidates are 40% of the score.** Getting the right ICD/RxNorm code is the
   single biggest lever, which is why the improved tier focuses there.

`score.py` implements a local read of this formula. Scoring the gold against
itself gives a perfect `100.00` (a good sanity check):

```bash
python score.py --pred data/dev/gold --gold data/dev
```

## Constraints

The competition is self-hosted and offline: no external LLM APIs, and every model
must be ≤ 9B parameters. That rules out GPT-4-style API calls and shapes the whole
solution around small, local, open models.
