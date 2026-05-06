#!/usr/bin/env python3
"""
Enumerate all set-theoretic prompts based on parent topics for each experiment.

For 2 topics (binding_set, support): 3^2 - 1 = 8 combinations.
For 3 topics (feasibility): 3^3 - 1 = 26 combinations.
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

MODEL_TAG = {
    "gpt-4o-mini": "4omini",
    "gpt-5.4":     "54",
    "gpt-5-mini":  "5mini",
}


def model_tag(model: str) -> str:
    return MODEL_TAG.get(model, model.replace(".", "").replace("-", ""))


def temp_tag(t):
    return "tempnone" if t is None else f"temp{t}".replace(".", "")


def file_tag(gen_model, gen_temp, cls_model, cls_temp):
    return (
        f"gen{model_tag(gen_model)}_gen{temp_tag(gen_temp)}"
        f"_cls{model_tag(cls_model)}_cls{temp_tag(cls_temp)}"
    )


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
                incl_op_sym = "∧" if incl_op else "∨"
                excl_op_sym = "∧" if excl_op else "∨"
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
                    label = f"{incl_lbl}∧¬{excl_lbl}"
                elif incl_keys:
                    name = incl_name
                    label = incl_label
                else:
                    excl_part = f"({excl_name})" if len(excl_keys) >= 2 else excl_name
                    name = f"NOT {excl_part}"
                    excl_lbl = f"({excl_label})" if len(excl_keys) >= 2 else excl_label
                    label = f"¬{excl_lbl}"

                combos.append({
                    "name": name, "label": label,
                    "inclusion": [topics[k] for k in incl_keys],
                    "exclusion": [topics[k] for k in excl_keys],
                    "inclusion_op": incl_op, "exclusion_op": excl_op,
                })

    return combos


def run_prompt(combo, client, classification_topics, num_seeds, list_length, gen_model, gen_temp, cls_model, cls_temp):
    """Run a single prompt combination across all seeds."""
    prompt = generate_inclusion_exclusion_prompt(
        inclusion_topics=combo["inclusion"], exclusion_topics=combo["exclusion"],
        inclusion_op=combo["inclusion_op"], exclusion_op=combo["exclusion_op"],
        n=list_length)

    seeds_data = []
    dims = list(classification_topics.keys())

    for seed in range(1, num_seeds + 1):
        try:
            papers = get_papers(prompt, client, seed, gen_model, gen_temp)
            vector = classify_paper_list_batch(papers, classification_topics, client, seed, cls_model, cls_temp)
            seeds_data.append({"papers": papers, "vector": vector})
            print(f"    Seed {seed}/{num_seeds}: {len(papers)} papers -> {format_vector(vector)}")
        except Exception as e:
            print(f"    Seed {seed}/{num_seeds}: ERROR - {e}")
            time.sleep(2)

    if seeds_data:
        mean_vec, std_vec = {}, {}
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
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Generator model")
    parser.add_argument("--temperature", "-t", default="0", help="Generation temperature (use 'none' to omit)")
    parser.add_argument("--cls-model", default=None, help="Classification model (defaults to --model)")
    parser.add_argument("--cls-temperature", default=None, help="Classification temperature (use 'none' to omit; defaults to --temperature)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    def parse_temp(val):
        if val is None or val == "" or str(val).lower() == "none":
            return None
        return float(val)

    gen_model = args.model
    gen_temp = parse_temp(args.temperature)
    cls_model = args.cls_model if args.cls_model is not None else gen_model
    cls_temp = parse_temp(args.cls_temperature) if args.cls_temperature is not None else gen_temp

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generator: {gen_model} @ {gen_temp}, Judge: {cls_model} @ {cls_temp}, "
          f"Seeds: {args.seeds}, List length: {args.list_length}")

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

        tag = file_tag(gen_model, gen_temp, cls_model, cls_temp)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DATA_DIR / f"{exp_name}_enumerated_{tag}.json"
        partial_file = DATA_DIR / f"{exp_name}_enumerated_{tag}_partial.json"

        config = {
            "model": gen_model, "temperature": gen_temp,
            "classification_model": cls_model, "classification_temperature": cls_temp,
            "num_seeds": args.seeds, "list_length": args.list_length,
            "timestamp": timestamp, "experiment": exp_name,
        }

        results = []
        for i, combo in enumerate(combos):
            print(f"\n[{i + 1}/{len(combos)}] {combo['name']} ({combo['label']})")
            result = run_prompt(combo, client, class_topics, args.seeds, args.list_length,
                                gen_model, gen_temp, cls_model, cls_temp)
            if result:
                results.append(result)

            partial_output = {
                "config": {**config, "completed_combos": i + 1, "total_combos": len(combos)},
                "parent_topics": parent_topics,
                "classification_topics": {k: v for k, v in class_topics.items()},
                "prompts": results,
            }
            partial_file.write_text(json.dumps(partial_output, indent=2))

        output = {
            "config": config,
            "parent_topics": parent_topics,
            "classification_topics": {k: v for k, v in class_topics.items()},
            "prompts": results,
        }
        output_file.write_text(json.dumps(output, indent=2))
        if partial_file.exists():
            partial_file.unlink()
        print(f"\nSaved to: {output_file}")

    print(f"\n{'=' * 70}\nALL EXPERIMENTS COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
