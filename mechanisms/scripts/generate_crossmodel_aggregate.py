#!/usr/bin/env python3
"""
Cross-model aggregation from existing experiment data.

Samples L_A from model m1 and L_B from model m2 (and vice versa),
computes intersection/union, and classifies with a fixed judge model.
No new generation calls needed — only classification of aggregated lists.

Usage:
    export OPENAI_API_KEY="<your-api-key>"
    python generate_crossmodel_from_existing.py \
        --m1-tag gen4omini_gentemp07_cls4omini_clstemp00 \
        --m2-tag gen54_gentemp00_cls4omini_clstemp00 \
        --cls-model gpt-4o-mini --cls-temperature 0 \
        --seeds 30
"""

import os
import sys
import json
import random
import argparse
import statistics
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_client, classify_paper_list_batch, find_intersection, find_union
from config import (
    BINDING_SET_CHILD_TOPICS,
    FEASIBILITY_TOPICS,
    SUPPORT_CHILD_TOPICS,
    LIST_LABELS,
)


EXPERIMENT_DEFS = {
    "binding_set": {
        "topics": BINDING_SET_CHILD_TOPICS,
        "L_A": "L_y1",
        "L_B": "L_y2",
        "agg_op": "intersection",
        "agg_name": "L_intersection",
        "focal_name": "L_x1_or",
    },
    "feasibility": {
        "topics": FEASIBILITY_TOPICS,
        "L_A": "L2",
        "L_B": "L3",
        "agg_op": "intersection",
        "agg_name": "L4",
        "focal_name": "L1",
    },
    "support": {
        "topics": SUPPORT_CHILD_TOPICS,
        "L_A": "L2_y1",
        "L_B": "L2_y2",
        "agg_op": "union",
        "agg_name": "L2",
        "focal_name": "L1",
    },
}


def aggregate_vectors(vectors: List[Dict[str, float]]) -> Dict[str, Dict]:
    if not vectors:
        return {}
    keys = vectors[0].keys()
    result = {}
    for key in keys:
        values = [v[key] for v in vectors]
        result[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return result


def load_aggregate_file(data_dir, exp, tag):
    """Load aggregate file and return list of seed data."""
    path = os.path.join(data_dir, f"{exp}_aggregate_{tag}.json")
    with open(path) as f:
        data = json.load(f)
    return data["seeds"]


def run_crossmodel_experiment(exp_name, m1_seeds, m2_seeds, exp_def, n_seeds,
                               client, cls_model, cls_temperature, rng, log):
    """Run cross-model aggregation for one experiment, one ordering."""
    topics = exp_def["topics"]
    la_key = exp_def["L_A"]
    lb_key = exp_def["L_B"]
    agg_op = find_intersection if exp_def["agg_op"] == "intersection" else find_union
    agg_name = exp_def["agg_name"]
    focal_name = exp_def["focal_name"]

    seeds_data = []
    for i in range(n_seeds):
        # Sample one seed from each model
        s_a = rng.choice(m1_seeds)
        s_b = rng.choice(m2_seeds)

        papers_a = s_a[la_key]["papers"]
        papers_b = s_b[lb_key]["papers"]
        papers_agg = agg_op(papers_a, papers_b)

        # Also sample focal list from m1 (first model)
        papers_focal = s_a[focal_name]["papers"]

        log(f"  Seed {i+1}/{n_seeds}: {la_key}={len(papers_a)}, {lb_key}={len(papers_b)}, "
            f"{agg_name}={len(papers_agg)}")

        # Classify
        vec_a = classify_paper_list_batch(papers_a, topics, client, i + 1, cls_model, cls_temperature)
        vec_b = classify_paper_list_batch(papers_b, topics, client, i + 1, cls_model, cls_temperature)
        if papers_agg:
            vec_agg = classify_paper_list_batch(papers_agg, topics, client, i + 1, cls_model, cls_temperature)
        else:
            vec_agg = {k: 0.0 for k in topics}
        vec_focal = classify_paper_list_batch(papers_focal, topics, client, i + 1, cls_model, cls_temperature)

        seeds_data.append({
            la_key: {"papers": papers_a, "vector": vec_a},
            lb_key: {"papers": papers_b, "vector": vec_b},
            agg_name: {"papers": papers_agg, "vector": vec_agg},
            focal_name: {"papers": papers_focal, "vector": vec_focal},
        })

    # Compute aggregates
    list_names = [la_key, lb_key, agg_name, focal_name]
    aggregates = {}
    for ln in list_names:
        vectors = [s[ln]["vector"] for s in seeds_data]
        aggregates[ln] = aggregate_vectors(vectors)

    return seeds_data, aggregates


def main():
    parser = argparse.ArgumentParser(description="Cross-model aggregation from existing data")
    parser.add_argument("--experiment", choices=["binding_set", "feasibility", "support", "all"], default="all")
    parser.add_argument("--m1-tag", required=True, help="File tag for model 1 (e.g. gen4omini_gentemp07_cls4omini_clstemp00)")
    parser.add_argument("--m2-tag", required=True, help="File tag for model 2 (e.g. gen54_gentemp00_cls4omini_clstemp00)")
    parser.add_argument("--cls-model", default="gpt-4o-mini", help="Classification model")
    parser.add_argument("--cls-temperature", type=float, default=0, help="Classification temperature")
    parser.add_argument("--seeds", "-s", type=int, default=30, help="Number of cross-model seeds to generate")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    def log(msg):
        print(msg, flush=True)

    client = get_client(args.api_key)
    rng = random.Random(args.random_seed)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_data")
    experiments = list(EXPERIMENT_DEFS.keys()) if args.experiment == "all" else [args.experiment]

    # Extract short model tags from file tags (e.g. "gen4omini_gentemp07_cls4omini_clstemp00" -> "4omini")
    m1_model = args.m1_tag.split("_")[0].replace("gen", "")
    m2_model = args.m2_tag.split("_")[0].replace("gen", "")
    m1_temp = args.m1_tag.split("_")[1].replace("gentemp", "")
    m2_temp = args.m2_tag.split("_")[1].replace("gentemp", "")

    MODEL_SHORT = {"gpt-4o-mini": "4omini", "gpt-5.4": "54", "gpt-5-mini": "5mini"}
    cls_tag = MODEL_SHORT.get(args.cls_model, args.cls_model.replace(".", "").replace("-", ""))
    cls_temp_tag = str(args.cls_temperature).replace(".", "")

    for exp_name in experiments:
        exp_def = EXPERIMENT_DEFS[exp_name]

        log(f"\n{'=' * 70}")
        log(f"{exp_name.upper()}")
        log(f"{'=' * 70}")

        # Load existing data
        m1_seeds = load_aggregate_file(data_dir, exp_name, args.m1_tag)
        m2_seeds = load_aggregate_file(data_dir, exp_name, args.m2_tag)
        log(f"Loaded m1: {len(m1_seeds)} seeds, m2: {len(m2_seeds)} seeds")

        # Ordering A: L_A from m1, L_B from m2
        log(f"\nOrdering A: {exp_def['L_A']} from {m1_model}, {exp_def['L_B']} from {m2_model}")
        seeds_a, agg_a = run_crossmodel_experiment(
            exp_name, m1_seeds, m2_seeds, exp_def, args.seeds,
            client, args.cls_model, args.cls_temperature, rng, log)

        tag_a = f"genA{m1_model}_genB{m2_model}_gentemp{m1_temp}x{m2_temp}_cls{cls_tag}_clstemp{cls_temp_tag}"
        output_a = {
            "config": {
                "m1_tag": args.m1_tag, "m2_tag": args.m2_tag,
                "ordering": f"{exp_def['L_A']}={m1_model}, {exp_def['L_B']}={m2_model}",
                "classification_model": args.cls_model,
                "classification_temperature": args.cls_temperature,
                "num_seeds": args.seeds, "random_seed": args.random_seed,
            },
            "seeds": seeds_a, "aggregates": agg_a,
        }
        path_a = os.path.join(data_dir, f"{exp_name}_aggregate_{tag_a}.json")
        with open(path_a, 'w') as f:
            json.dump(output_a, f, indent=2)
        log(f"Saved to {os.path.basename(path_a)}")

        # Ordering B: L_A from m2, L_B from m1
        log(f"\nOrdering B: {exp_def['L_A']} from {m2_model}, {exp_def['L_B']} from {m1_model}")
        seeds_b, agg_b = run_crossmodel_experiment(
            exp_name, m2_seeds, m1_seeds, exp_def, args.seeds,
            client, args.cls_model, args.cls_temperature, rng, log)

        tag_b = f"genA{m2_model}_genB{m1_model}_gentemp{m2_temp}x{m1_temp}_cls{cls_tag}_clstemp{cls_temp_tag}"
        output_b = {
            "config": {
                "m1_tag": args.m1_tag, "m2_tag": args.m2_tag,
                "ordering": f"{exp_def['L_A']}={m2_model}, {exp_def['L_B']}={m1_model}",
                "classification_model": args.cls_model,
                "classification_temperature": args.cls_temperature,
                "num_seeds": args.seeds, "random_seed": args.random_seed,
            },
            "seeds": seeds_b, "aggregates": agg_b,
        }
        path_b = os.path.join(data_dir, f"{exp_name}_aggregate_{tag_b}.json")
        with open(path_b, 'w') as f:
            json.dump(output_b, f, indent=2)
        log(f"Saved to {os.path.basename(path_b)}")

        # Print summary
        list_names = [exp_def["L_A"], exp_def["L_B"], exp_def["agg_name"], exp_def["focal_name"]]
        log(f"\nSummary:")
        for ln in list_names:
            va = {k: f"{v['mean']:.3f}" for k, v in agg_a[ln].items()}
            vb = {k: f"{v['mean']:.3f}" for k, v in agg_b[ln].items()}
            log(f"  {ln}: A={va}  B={vb}")

    log("\nAll experiments complete.")


if __name__ == "__main__":
    main()
