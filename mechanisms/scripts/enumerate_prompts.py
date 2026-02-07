#!/usr/bin/env python3
"""
Enumerate all set-theoretic prompts based on parent topics for each experiment.

For 2 topics (binding_set, support): 3^2 - 1 = 8 combinations
For 3 topics (feasibility): 3^3 - 1 = 26 combinations
With AND/OR operator variants where applicable.
"""

import os
import sys
import json
import time
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from config import (
    BINDING_SET_PARENT_TOPICS,
    BINDING_SET_CHILD_TOPICS,
    FEASIBILITY_PARENT_TOPICS,
    FEASIBILITY_TOPICS,
    SUPPORT_PARENT_TOPICS,
    SUPPORT_CHILD_TOPICS,
)

from utils import (
    get_papers,
    classify_paper_list_batch,
    generate_inclusion_exclusion_prompt,
)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "experiment_data"

EXPERIMENTS = {
    "binding_set": {
        "parent_topics": BINDING_SET_PARENT_TOPICS,
        "classification_topics": BINDING_SET_CHILD_TOPICS,
    },
    "feasibility": {
        "parent_topics": FEASIBILITY_PARENT_TOPICS,
        "classification_topics": FEASIBILITY_TOPICS,
    },
    "support": {
        "parent_topics": SUPPORT_PARENT_TOPICS,
        "classification_topics": SUPPORT_CHILD_TOPICS,
    },
}


def format_vector(v: Dict[str, float]) -> str:
    return ", ".join([f"{k}={v:.2f}" for k, v in v.items()])


def generate_all_prompt_combos(topics: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate all 3^n - 1 set-theoretic prompt combinations.

    Each topic can be: included (1), excluded (2), or unmentioned (0).
    For groups of 2+ included or excluded, generates AND/OR variants.
    """
    keys = list(topics.keys())
    n = len(keys)
    combos = []

    for assignment in product(range(3), repeat=n):
        incl_keys = [keys[i] for i in range(n) if assignment[i] == 1]
        excl_keys = [keys[i] for i in range(n) if assignment[i] == 2]

        if not incl_keys and not excl_keys:
            continue

        incl_ops = [True, False] if len(incl_keys) >= 2 else [True]
        excl_ops = [True, False] if len(excl_keys) >= 2 else [True]

        for incl_op in incl_ops:
            for excl_op in excl_ops:
                incl_op_sym = "\u2227" if incl_op else "\u2228"
                excl_op_sym = "\u2227" if excl_op else "\u2228"
                incl_op_word = " AND " if incl_op else " OR "
                excl_op_word = " AND " if excl_op else " OR "

                incl_name = incl_op_word.join(incl_keys)
                excl_name = excl_op_word.join(excl_keys)
                incl_label = incl_op_sym.join(incl_keys)
                excl_label = excl_op_sym.join(excl_keys)

                if incl_keys and excl_keys:
                    incl_part = f"({incl_name})" if len(incl_keys) >= 2 else incl_name
                    excl_part = f"({excl_name})" if len(excl_keys) >= 2 else excl_name
                    name = f"{incl_part} AND NOT {excl_part}"
                    incl_lbl = f"({incl_label})" if len(incl_keys) >= 2 else incl_label
                    excl_lbl = f"({excl_label})" if len(excl_keys) >= 2 else excl_label
                    label = f"{incl_lbl}\u2227\u00ac{excl_lbl}"
                elif incl_keys:
                    name = incl_name
                    label = incl_label
                else:
                    excl_part = f"({excl_name})" if len(excl_keys) >= 2 else excl_name
                    name = f"NOT {excl_part}"
                    excl_lbl = f"({excl_label})" if len(excl_keys) >= 2 else excl_label
                    label = f"\u00ac{excl_lbl}"

                combos.append({
                    "name": name, "label": label,
                    "inclusion": [topics[k] for k in incl_keys],
                    "exclusion": [topics[k] for k in excl_keys],
                    "inclusion_op": incl_op, "exclusion_op": excl_op,
                })

    return combos


def run_prompt(combo, client, classification_topics, num_seeds, list_length=20):
    """Run a single prompt combination across all seeds."""
    prompt = generate_inclusion_exclusion_prompt(
        inclusion_topics=combo["inclusion"], exclusion_topics=combo["exclusion"],
        inclusion_op=combo["inclusion_op"], exclusion_op=combo["exclusion_op"],
        n=list_length)

    seeds_data = []
    dims = list(classification_topics.keys())

    for seed in range(1, num_seeds + 1):
        try:
            papers = get_papers(prompt, client, seed, MODEL, TEMPERATURE)
            vector = classify_paper_list_batch(papers, classification_topics, client, seed, MODEL, TEMPERATURE)
            seeds_data.append({"papers": papers, "vector": vector})
            print(f"    Seed {seed}/{num_seeds}: {len(papers)} papers -> {format_vector(vector)}")
        except Exception as e:
            print(f"    Seed {seed}/{num_seeds}: ERROR - {e}")
            time.sleep(2)

    if seeds_data:
        mean_vec = {}
        std_vec = {}
        for d in dims:
            values = [s["vector"].get(d, 0.0) for s in seeds_data]
            mean_vec[d] = statistics.mean(values)
            std_vec[d] = statistics.stdev(values) if len(values) > 1 else 0.0
        return {
            "name": combo["name"], "label": combo["label"],
            "num_seeds": len(seeds_data), "mean": mean_vec, "std": std_vec,
            "seeds": seeds_data,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Enumerate all set-theoretic prompts")
    parser.add_argument("--experiment", choices=["binding_set", "feasibility", "support", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--list-length", type=int, default=20)
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Model: {MODEL}, Seeds: {args.seeds}, List length: {args.list_length}")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set OPENAI_API_KEY")
        return
    client = OpenAI(api_key=api_key, timeout=120)

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    for exp_name in experiments:
        exp = EXPERIMENTS[exp_name]
        parent_topics = exp["parent_topics"]
        class_topics = exp["classification_topics"]

        print(f"\n{'=' * 70}\nEXPERIMENT: {exp_name.upper()}\n{'=' * 70}")

        combos = generate_all_prompt_combos(parent_topics)
        print(f"Total prompt combinations: {len(combos)}")

        results = []
        for i, combo in enumerate(combos):
            print(f"\n[{i + 1}/{len(combos)}] {combo['name']} ({combo['label']})")
            result = run_prompt(combo, client, class_topics, args.seeds, args.list_length)
            if result:
                results.append(result)

        output = {
            "config": {
                "model": MODEL, "temperature": TEMPERATURE,
                "num_seeds": args.seeds, "list_length": args.list_length,
                "timestamp": timestamp, "experiment": exp_name,
            },
            "parent_topics": parent_topics,
            "classification_topics": {k: v for k, v in class_topics.items()},
            "prompts": results,
        }

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DATA_DIR / f"{exp_name}_enumerated_{timestamp}.json"
        output_file.write_text(json.dumps(output, indent=2))
        print(f"\nSaved to: {output_file}")

    print(f"\n{'=' * 70}\nALL EXPERIMENTS COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
