"""Analysis helpers for aggregation experiments."""

import math
from typing import Tuple
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ExperimentConfig:
    name: str
    input_topics: Dict[str, str]
    output_dims: Dict[str, str]
    L_A_desc: str
    L_B_desc: str
    L_f_desc: str


EXPERIMENTS = {
    "binding_set": ExperimentConfig(
        name="Binding Set Contraction",
        input_topics={"y1": "NLP", "y2": "Computer Vision"},
        output_dims={"x1": "deep learning", "x2": "old NLP", "x3": "old CV", "x4": "multimodal", "x5": "statistical ML"},
        L_A_desc="NLP", L_B_desc="CV", L_f_desc="deep learning (excl)",
    ),
    "feasibility": ExperimentConfig(
        name="Feasibility Expansion",
        input_topics={"x1": "blockchain", "x2": "cryptography", "x3": "distributed systems"},
        output_dims={"x1": "blockchain", "x2": "cryptography", "x3": "distributed systems"},
        L_A_desc="x1\u2228x2 (excl x3)", L_B_desc="x1\u2228x3 (excl x2)", L_f_desc="x1 (excl x2\u2228x3)",
    ),
    "support": ExperimentConfig(
        name="Support Expansion",
        input_topics={"y1": "theoretical CS", "y2": "economics"},
        output_dims={"x1": "complexity theory", "x2": "macroeconomics", "x3": "mechanism design"},
        L_A_desc="theoretical CS", L_B_desc="economics", L_f_desc="x1\u2228x2 prompt",
    ),
}

DIRECTIONAL_LABELS = {
    "binding_set": [
        ("L_y1_and_y2", "DIR: y1 \u2229 y2"),
        ("L_excl_y1_and_y2", "DIR: \u00ac(y1 \u2229 y2)"),
        ("L_excl_y1_or_y2", "DIR: \u00ac(y1 \u222a y2)"),
        ("L_y1_not_y2", "DIR: y1 \u2227 \u00acy2"),
    ],
    "feasibility": [
        ("L1_and_excl", "DIR: x1\u2227\u00ac(x2\u2227x3)"),
        ("L1_or_excl", "DIR: x1\u2227\u00ac(x2\u2228x3)"),
    ],
    "support": [
        ("L_y1_and_y2", "DIR: y1 \u2229 y2"),
        ("L_y1_or_y2", "DIR: y1 \u222a y2"),
    ],
}


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    if p <= 0:
        upper = 1 - (0.05) ** (1/n) if n > 0 else 0.0
        return (0.0, min(1.0, upper))
    if p >= 1:
        lower = (0.05) ** (1/n) if n > 0 else 1.0
        return (max(0.0, lower), 1.0)
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    variance = p*(1-p)/n + z**2/(4*n**2)
    if variance < 0:
        variance = 0
    spread = z * math.sqrt(variance) / denominator
    return (max(0, center - spread), min(1, center + spread))
