# Power and Limitations of Aggregation in Compound AI Systems

Code and data for the empirical sections of the paper *Power and Limitations of
Aggregation in Compound AI Systems*.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Experiments

Three set-theoretic experiments are defined in `mechanisms/config.py`:

- **Binding-set contraction** — `L_A ∩ L_B` over two parent topics (NLP, CV).
- **Feasibility expansion** — pairwise unions over three parent topics
  (blockchain, cryptography, distributed systems).
- **Support expansion** — `L_A ∪ L_B` over two parent topics (theoretical CS,
  economics).

Each experiment generates paper lists from an LLM and classifies each list
along a fixed set of output dimensions to obtain probability vectors.

## Single-model run

Generate the four core lists (`L_A`, `L_B`, `L_A_B_agg`, `L_f`) for all three
experiments:

```bash
python mechanisms/scripts/generate_aggregate.py \
    --experiment all \
    --model gpt-4o-mini --temperature 0.7 \
    --cls-model gpt-4o-mini --cls-temperature 0 \
    --seeds 30 --list-length 20
```

Enumerate every set-theoretic prompt over the parent topics:

```bash
python mechanisms/scripts/enumerate_prompts.py \
    --experiment all \
    --model gpt-4o-mini --temperature 0.7 \
    --cls-model gpt-4o-mini --cls-temperature 0 \
    --seeds 30 --list-length 20
```

Pass `--temperature none` (or `--cls-temperature none`) to omit the
temperature parameter entirely — required for models that don't accept it.

## Cross-model run

Once two single-model aggregate runs exist, mix their lists and reclassify
with a fixed judge model:

```bash
python mechanisms/scripts/generate_crossmodel_aggregate.py \
    --experiment all \
    --m1-tag gen4omini_gentemp00_cls4omini_clstemp00 \
    --m2-tag gen54_gentemp00_cls4omini_clstemp00 \
    --cls-model gpt-4o-mini --cls-temperature 0 \
    --seeds 30 --random-seed 42
```

This produces both orderings of (L_A from model 1, L_B from model 2) and
(L_A from model 2, L_B from model 1).

## Data layout

Outputs live in `mechanisms/experiment_data/`:

- **Single-model**:
  `{experiment}_{kind}_gen{MODEL}_gentemp{TEMP}_cls{MODEL}_clstemp{TEMP}.json`
- **Cross-model**:
  `{experiment}_aggregate_genA{A}_genB{B}_gentemp{tA}x{tB}_cls{C}_clstemp{tC}.json`

`{kind}` is `aggregate` or `enumerated`. Model and temperature tags use the
short forms below:

| Model | Tag | Temperatures available |
|---|---|---|
| gpt-4o-mini | `4omini` | `00`, `03`, `05`, `07` |
| gpt-5.4     | `54`     | `00`, `07` |
| gpt-5-mini  | `5mini`  | `none` |

All ship with the judge fixed to `gpt-4o-mini` at temperature 0. Cross-model
data covers all six pairings of `{gpt-4o-mini, gpt-5.4, gpt-5-mini}` at
default temperatures.

## Analysis

Open `mechanisms/mechanisms.ipynb`, set the parameters in the first code
cell, and run all cells. The notebook supports two modes:

- `MODE = "single"` — one generator, one judge. Loads the aggregate +
  enumerated files for that configuration, prints Wilson CI tables, finds
  P_agg (the enumerated prompt closest to L_A_B_agg in CI-box L1 distance),
  and produces 2D projection plots with Wilson CI rectangles.
- `MODE = "cross"` — two generators (M1 and M2) plus a judge. Loads both
  orderings of (L_A, L_B), prints CI tables for each, and searches P_agg
  over the union of enumerated prompts from both models. Produces two
  plots per experiment (one per ordering).

Plots are saved to `mechanisms/figures/`. The default-config (single-mode,
`gpt-4o-mini @ 0.7`) PNGs ship pre-rendered there; running the notebook
overwrites them with whatever configuration is currently set.

For the rich-embedding analysis of LLM aggregation, see
[`rich_embeddings/README.md`](rich_embeddings/README.md).
