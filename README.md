# Power and Limitations of Aggregation in Compound AI Systems

Code and data for the empirical sections of the paper *Power and Limitations of
Aggregation in Compound AI Systems*.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Experiments

- **[`mechanisms/`](mechanisms/README.md)** — set-theoretic aggregation
  experiments (binding-set contraction, feasibility expansion, support
  expansion) over LLM-generated paper lists.
- **[`rich_embeddings/`](rich_embeddings/README.md)** — sentence-transformer
  embedding analysis of union- vs intersection-style LLM aggregations on a
  reference-generation task, with L2-distance heatmap and 2D/3D
  task-specific projection plots.
