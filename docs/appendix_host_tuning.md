# Appendix: domain tuning

The baseline and improved tiers are deliberately general — they teach methods you
would reuse on any clinical-IE task. The last point or two on the leaderboard,
though, came from **competition-specific tuning**: reverse-engineering the exact
linking conventions the organizer's gold labels follow. This is worth understanding
as an example of how leaderboard optimization differs from general method, but it is
*not* something to generalise from, so it lives here rather than in the core code.

None of these are enabled in the shipped configs.

## Why it's tricky

The organizer never published the full linking rules, and their few worked examples
sometimes contradict a simple rule. So the conventions can only be inferred from
(a) the handful of published examples and (b) leaderboard feedback — each
submission is one bit of information. That makes it slow, empirical tuning, not a
principled model.

## The levers that helped

1. **A better ICD knowledge base.** Adding the Vietnamese Thông tư 06 / QĐ 4469 code
   list (with synonyms) on top of the WHO English list improved diagnosis linking —
   more of the gold codes were actually present to be retrieved. This one is just
   "better data" and is the least competition-specific of the lot.

2. **ICD leaf preference.** Gold ICD codes are almost always specific leaves
   (`E11.9`), never bare categories (`E11`). Dropping the bare category from the
   shortlist, and defaulting to the `.9` ("unspecified") leaf when no complication
   is described, matched the gold convention. (A light version of this is kept in
   the improved reranker.)

3. **Dose-range → lower bound.** For "acetaminophen 325-650 mg", the gold code is
   the 325 mg product, not 650 — the organizer takes the lower bound. A one-line
   mention normalization ("N–M unit" → "N unit") captured that.

4. **Strength-less mention → ingredient.** When a drug is named without a strength
   ("nystatin oral suspension"), the gold sometimes links to the ingredient concept
   rather than a specific product. This lever helped on some examples but is
   genuinely ambiguous — other published examples point the other way — so it is a
   contested convention, not a reliable rule.

## The lesson

Levers 1–2 are close to general good practice (better data, respect the label
granularity). Levers 3–4 are pure convention-matching: they only help because they
mirror one specific grader, and getting them wrong can *lose* points. On a real
project you would spend that effort on the model, not on guessing a rubric — which
is exactly why the teaching tiers stop before them.
