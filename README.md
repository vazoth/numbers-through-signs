# Numbers Through Signs

Data and measurement code for *Numbers Through Signs: How LLMs Connect Counts,
Number Words, and Numerals* (MathNLP 2026), by J. Albert Yoo,
Eötvös Loránd University.

## Analysis

With Python 3.11 or 3.12:

```sh
python -m pip install -r requirements.txt
python analyze.py
```

This analyzes stored measurements and writes to `results/`:

- `sites.csv`, `admission.csv`, `readouts.csv`: selection, screened wordings, correlations.
- `one_shot.csv`: direct interventions and same-number `identity` preservation;
  interpretation requires both `admitted` and `meets_direct_criterion`.
- `interchange.csv`, `depth.csv`, `subspace.csv`: grouped intervention measurements.

IIA uses eligible trials; gain uses all requested trials. `preservation`
means target-answer correctness. Wrong-value donors need not change the answer;
their IIA/gain remain oriented to the designated source. The requirements cover stored-record analysis;
measurement source is provided for inspection, without activation caches.

`data/` contains trial records, readout predictions, representation inputs,
and fitted subspace bases. Measurement rows link through `record_id` to
`runs.json`; representation inputs use `row_index`. `protocol.json` gives
selection rules and thresholds. `methods/numbers_through_signs/` contains measurement
code; `configs/` specifies scientific settings and checkpoint revisions.

The paper describes cross-format site selection. The implemented selector
pools all four digit/word directions for 0.6B–4B, including same-format pairs;
8B, 14B and 30B-A3B use only the two cross-format directions.
Selection uses development to-end trajectories. Admission and intervention
use separate BF16 baselines, so eligible-trial counts can differ. Exact-target
controls use a same-batch reference in `exact_noop_margin_delta`.

## Comparison limitation

Comparison uses `value > threshold`, labeling false cases—including equal values—as
“smaller”. Reported results include ties; excluding them changes site selection
and causes 14B comparison to fail the screening threshold.

## Paper

`Numbers_Through_Signs_MathNLP_camera_ready.pdf` is the published paper.
To compile the source with a TeX distribution and `latexmk`:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

See `LICENSE` and `THIRD_PARTY_NOTICES.md` for licensing.
