# 3. The improvement: retrieve-then-rerank

The baseline's weakest step is linking. SapBERT retrieval is fast and gets the
correct code into the top-k, but it ranks by raw string similarity, so the exact
answer often sits at rank 2–5 rather than rank 1 — especially when several codes
look alike ("Metoprolol Tartrate 25 MG" vs "Metoprolol Succinate 25 MG ER").

## The idea

Split the job into two stages, a standard pattern in modern retrieval:

```
mention ──► retriever (top-k, cheap, high recall) ──► LLM reranker (picks the one)
```

The retriever proposes a shortlist; a small **local LLM** reads the note and the
shortlist and picks the code(s) that actually fit. Crucially, the LLM is
**constrained to the retrieved codes** — it chooses from the list, so it can never
hallucinate an invalid code. If it isn't confident, it returns nothing (better than
forcing a wrong code, given the double penalty).

Config: [`configs/improved.yaml`](../configs/improved.yaml) — it `extends`
`baseline.yaml` and adds one block:

```yaml
normalization:
  llm_rerank:
    retrieve_k: 20            # hand the LLM a wider shortlist
```

Implementation: `medextract/normalization/llm_reranker.py`. It also drops a bare
ICD category (e.g. `E11`) from the shortlist when a specific sub-code is present,
nudging the LLM toward a real leaf code.

## The model, and running it on a free GPU

The default LLM is **Qwen3-8B** (8.2B ≤ 9B, competition-legal, fully local). At
full precision it needs ~18 GB. To make the demo runnable for everyone,
`configs/improved.yaml` sets `load_in_4bit: true`, which quantizes the weights to
4-bit (nf4) so the model fits in **~6 GB** — inside a free Colab T4's budget.

For the actual submission run on a bigger GPU, flip it back to full precision:

```yaml
llm:
  load_in_4bit: false
  device: wait
  min_free_gb: 18.0
```

## Run it

```bash
# dev
python run.py --config configs/improved.yaml --input data/dev/input --output out/dev_imp
python score.py --pred out/dev_imp --gold data/dev

# a submission zip on the real test set
python run.py --config configs/improved.yaml --input <test dir> --output out/sub --zip
```

The reranker lifts the candidate score (host ≈ 24.5–26 vs the baseline's ≈ 21.8)
while leaving NER and assertions unchanged. From here, the remaining gains come
from domain-specific tuning — see [the appendix](appendix_host_tuning.md).
