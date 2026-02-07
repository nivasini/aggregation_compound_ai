#!/usr/bin/env python3
"""
Generate aggregate prompts (L_A, L_B, L_A_B_agg, L_f) for all 3 experiments.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python generate_aggregate.py --seeds 10

API calls per seed: 8 (binding) + 7 (feasibility) + 7 (support) = 22
"""

import os
import sys
import json
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_client,
    get_papers,
    classify_paper_list_batch,
    find_intersection,
    find_union,
    generate_inclusion_exclusion_prompt,
)

from config import (
    BINDING_SET_PARENT_TOPICS,
    BINDING_SET_CHILD_TOPICS,
    FEASIBILITY_TOPICS,
    SUPPORT_PARENT_TOPICS,
    SUPPORT_CHILD_TOPICS,
)


def format_vector(vec: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted(vec.items()))


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


def run_binding_set_seed(seed, client, model, temperature, list_length, verbose=True):
    if verbose:
        print(f"  Generating L_y1 (NLP papers)...")
    prompt_y1 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[BINDING_SET_PARENT_TOPICS["y1"]], exclusion_topics=[], n=list_length)
    papers_y1 = get_papers(prompt_y1, client, seed, model, temperature)

    if verbose:
        print(f"  Generating L_y2 (CV papers)...")
    prompt_y2 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[BINDING_SET_PARENT_TOPICS["y2"]], exclusion_topics=[], n=list_length)
    papers_y2 = get_papers(prompt_y2, client, seed, model, temperature)

    papers_intersection = find_intersection(papers_y1, papers_y2)
    if verbose:
        print(f"  L_intersection: {len(papers_intersection)} papers")

    if verbose:
        print(f"  Generating L_x1_or (direct deep learning)...")
    prompt_x1 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[BINDING_SET_CHILD_TOPICS["x1"]],
        exclusion_topics=[BINDING_SET_CHILD_TOPICS[k] for k in ["x2", "x3", "x4", "x5"]],
        inclusion_op=True, exclusion_op=False, n=list_length)
    papers_x1 = get_papers(prompt_x1, client, seed, model, temperature)

    if verbose:
        print(f"  Classifying lists (batch)...")
    vector_y1 = classify_paper_list_batch(papers_y1, BINDING_SET_CHILD_TOPICS, client, seed, model, temperature)
    vector_y2 = classify_paper_list_batch(papers_y2, BINDING_SET_CHILD_TOPICS, client, seed, model, temperature)
    vector_intersection = classify_paper_list_batch(papers_intersection, BINDING_SET_CHILD_TOPICS, client, seed, model, temperature) if papers_intersection else {k: 0.0 for k in BINDING_SET_CHILD_TOPICS}
    vector_x1 = classify_paper_list_batch(papers_x1, BINDING_SET_CHILD_TOPICS, client, seed, model, temperature)

    if verbose:
        for name, vec in [("L_y1", vector_y1), ("L_y2", vector_y2), ("L_intersection", vector_intersection), ("L_x1_or", vector_x1)]:
            print(f"    {name}: {format_vector(vec)}")

    return {
        "L_y1": {"papers": papers_y1, "vector": vector_y1},
        "L_y2": {"papers": papers_y2, "vector": vector_y2},
        "L_intersection": {"papers": papers_intersection, "vector": vector_intersection},
        "L_x1_or": {"papers": papers_x1, "vector": vector_x1},
    }


def run_feasibility_seed(seed, client, model, temperature, list_length, verbose=True):
    if verbose:
        print(f"  Generating L1 (blockchain, excl crypto OR dist sys)...")
    prompt_l1 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[FEASIBILITY_TOPICS["x1"]],
        exclusion_topics=[FEASIBILITY_TOPICS["x2"], FEASIBILITY_TOPICS["x3"]],
        inclusion_op=True, exclusion_op=False, n=list_length)
    papers_l1 = get_papers(prompt_l1, client, seed, model, temperature)

    if verbose:
        print(f"  Generating L2 (blockchain OR crypto, excl dist sys)...")
    prompt_l2 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[FEASIBILITY_TOPICS["x1"], FEASIBILITY_TOPICS["x2"]],
        exclusion_topics=[FEASIBILITY_TOPICS["x3"]],
        inclusion_op=False, exclusion_op=True, n=list_length)
    papers_l2 = get_papers(prompt_l2, client, seed, model, temperature)

    if verbose:
        print(f"  Generating L3 (blockchain OR dist sys, excl crypto)...")
    prompt_l3 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[FEASIBILITY_TOPICS["x1"], FEASIBILITY_TOPICS["x3"]],
        exclusion_topics=[FEASIBILITY_TOPICS["x2"]],
        inclusion_op=False, exclusion_op=True, n=list_length)
    papers_l3 = get_papers(prompt_l3, client, seed, model, temperature)

    papers_l4 = find_intersection(papers_l2, papers_l3)
    if verbose:
        print(f"  L4 (L2 ∩ L3): {len(papers_l4)} papers")

    if verbose:
        print(f"  Classifying lists (batch)...")
    vector_l1 = classify_paper_list_batch(papers_l1, FEASIBILITY_TOPICS, client, seed, model, temperature)
    vector_l2 = classify_paper_list_batch(papers_l2, FEASIBILITY_TOPICS, client, seed, model, temperature)
    vector_l3 = classify_paper_list_batch(papers_l3, FEASIBILITY_TOPICS, client, seed, model, temperature)
    vector_l4 = classify_paper_list_batch(papers_l4, FEASIBILITY_TOPICS, client, seed, model, temperature) if papers_l4 else {k: 0.0 for k in FEASIBILITY_TOPICS}

    if verbose:
        for name, vec in [("L1", vector_l1), ("L2", vector_l2), ("L3", vector_l3), ("L4", vector_l4)]:
            print(f"    {name}: {format_vector(vec)}")

    return {
        "L1": {"papers": papers_l1, "vector": vector_l1},
        "L2": {"papers": papers_l2, "vector": vector_l2},
        "L3": {"papers": papers_l3, "vector": vector_l3},
        "L4": {"papers": papers_l4, "vector": vector_l4},
    }


def run_support_seed(seed, client, model, temperature, list_length, verbose=True):
    if verbose:
        print(f"  Generating L1 (complexity theory OR macroeconomics)...")
    prompt_l1 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[SUPPORT_CHILD_TOPICS["x1"], SUPPORT_CHILD_TOPICS["x2"]],
        exclusion_topics=[], inclusion_op=False, n=list_length)
    papers_l1 = get_papers(prompt_l1, client, seed, model, temperature)

    if verbose:
        print(f"  Generating L2_y1 (theoretical CS)...")
    prompt_l2_y1 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[SUPPORT_PARENT_TOPICS["y1"]], exclusion_topics=[], n=list_length)
    papers_l2_y1 = get_papers(prompt_l2_y1, client, seed, model, temperature)

    if verbose:
        print(f"  Generating L2_y2 (economics)...")
    prompt_l2_y2 = generate_inclusion_exclusion_prompt(
        inclusion_topics=[SUPPORT_PARENT_TOPICS["y2"]], exclusion_topics=[], n=list_length)
    papers_l2_y2 = get_papers(prompt_l2_y2, client, seed, model, temperature)

    papers_l2 = find_union(papers_l2_y1, papers_l2_y2)
    if verbose:
        print(f"  L2 (union): {len(papers_l2)} papers")

    if verbose:
        print(f"  Classifying lists (batch)...")
    vector_l1 = classify_paper_list_batch(papers_l1, SUPPORT_CHILD_TOPICS, client, seed, model, temperature)
    vector_l2_y1 = classify_paper_list_batch(papers_l2_y1, SUPPORT_CHILD_TOPICS, client, seed, model, temperature)
    vector_l2_y2 = classify_paper_list_batch(papers_l2_y2, SUPPORT_CHILD_TOPICS, client, seed, model, temperature)
    vector_l2 = classify_paper_list_batch(papers_l2, SUPPORT_CHILD_TOPICS, client, seed, model, temperature)

    if verbose:
        for name, vec in [("L1", vector_l1), ("L2_y1", vector_l2_y1), ("L2_y2", vector_l2_y2), ("L2", vector_l2)]:
            print(f"    {name}: {format_vector(vec)}")

    return {
        "L1": {"papers": papers_l1, "vector": vector_l1},
        "L2_y1": {"papers": papers_l2_y1, "vector": vector_l2_y1},
        "L2_y2": {"papers": papers_l2_y2, "vector": vector_l2_y2},
        "L2": {"papers": papers_l2, "vector": vector_l2},
    }


def main():
    parser = argparse.ArgumentParser(description="Run aggregate prompts with batch classification")
    parser.add_argument("--seeds", "-s", type=int, default=10)
    parser.add_argument("--list-length", "-n", type=int, default=10)
    parser.add_argument("--model", "-m", default="gpt-4o-mini")
    parser.add_argument("--temperature", "-t", type=float, default=0)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--log", "-l", default=None)
    args = parser.parse_args()

    log_file = open(args.log, "w") if args.log else None

    def log(msg):
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    client = get_client(args.api_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_data")
    os.makedirs(output_dir, exist_ok=True)

    log(f"\nModel: {args.model}, Seeds: {args.seeds}, List length: {args.list_length}")
    log(f"Output dir: {output_dir}")

    config = {
        "model": args.model, "temperature": args.temperature,
        "num_seeds": args.seeds, "list_length": args.list_length,
        "timestamp": timestamp, "batch_classification": True,
    }

    EXPERIMENT_DEFS = {
        "binding_set": {
            "runner": run_binding_set_seed,
            "list_names": ["L_y1", "L_y2", "L_intersection", "L_x1_or"],
        },
        "feasibility": {
            "runner": run_feasibility_seed,
            "list_names": ["L1", "L2", "L3", "L4"],
        },
        "support": {
            "runner": run_support_seed,
            "list_names": ["L1", "L2_y1", "L2_y2", "L2"],
        },
    }

    for exp_name, exp_def in EXPERIMENT_DEFS.items():
        log(f"\n{'=' * 70}\n{exp_name.upper()}\n{'=' * 70}")

        seeds = []
        for seed in range(1, args.seeds + 1):
            log(f"\n--- Seed {seed}/{args.seeds} ---")
            seeds.append(exp_def["runner"](seed, client, args.model, args.temperature, args.list_length))

        aggregates = {}
        for ln in exp_def["list_names"]:
            vectors = [s[ln]["vector"] for s in seeds]
            aggregates[ln] = aggregate_vectors(vectors)

        output = {"config": config, "seeds": seeds, "aggregates": aggregates}
        output_file = os.path.join(output_dir, f"{exp_name}_aggregate.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        log(f"Saved to {output_file}")

    log("\nAll experiments complete.")
    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
