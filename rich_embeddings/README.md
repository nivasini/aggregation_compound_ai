# Rich Embedding Analysis

Sentence-transformer embeddings of LLM outputs for the reference-generation task,
used to visualize how union-style and intersection-style aggregation transform
outputs in semantic space.

## What this directory contains

| File | Description |
|---|---|
| `embeddings.ipynb` | Self-contained analysis notebook |
| `papers_custom_outputs.json` | 7 LLM outputs: 5 from per-perspective prompts + 1 union aggregation + 1 intersection aggregation |
| `model_outputs_gpt_mini.json` | 805 background outputs from `gpt-4o-mini` on a fixed instruction set |
| `embeddings_gpt.npy` | Cached `(805, 768)` background embeddings (`sentence-transformers/all-mpnet-base-v2`) |

## Setup

Beyond the project root `requirements.txt`, the notebook needs:

```bash
pip install numpy pandas matplotlib seaborn scipy sentence-transformers
```

The notebook auto-installs an OpenAI client at runtime; export your key first:

```bash
export OPENAI_API_KEY="sk-..."
```

## What the notebook does

1. **Generate component outputs.** Sends 5 per-perspective prompts about
   "influential LLM papers" to `gpt-4o-mini` (machine-learning theory, NLP /
   computational linguistics, cognitive science, AI alignment, multi-agent /
   game theory).
2. **Generate aggregations.** Sends two follow-up prompts that ask the model
   to produce a 10-paper list reflecting the **union** of the five component
   lists, and another reflecting their **intersection**. Both ask for a
   freshly-generated list (not a literal set operation).
3. **Embed everything.** Encodes the 7 paper-task outputs and the 805
   background outputs with `sentence-transformers/all-mpnet-base-v2`
   (`d=768`).
4. **Visualize.** Three figures:
   - **Pairwise L2 heatmap** between the 7 paper-task outputs.
   - **2D projection** of the 7 outputs onto the two embedding dimensions of
     highest variance among the 7 outputs, with the 805 background
     embeddings shown in light gray.
   - **3D projection** onto the top-3 task-specific dimensions, same
     overlay.

Step 1 caches to `papers_custom_outputs.json` and step 3's background
embeddings cache to `embeddings_gpt.npy` — both files ship pre-computed, so
the notebook will skip generation/embedding if they exist. Delete the file
to force regeneration.

## Reproducing the figures

```bash
jupyter notebook rich_embeddings/embeddings.ipynb
```

Run all cells. With the cached files present this is a CPU-only computation
that completes in seconds (no API calls).
