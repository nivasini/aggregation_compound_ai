"""Shared configuration for aggregation experiments."""

# =============================================================================
# TOPIC DEFINITIONS
# =============================================================================

# Binding Set Contraction
BINDING_SET_PARENT_TOPICS = {
    "y1": "natural language processing (NLP)",
    "y2": "computer vision (CV)",
}

BINDING_SET_CHILD_TOPICS = {
    "x1": "deep learning (transformers, attention mechanisms, deep neural networks, modern architectures like BERT, GPT, ViT)",
    "x2": "non-transformer NLP methods (RNN, LSTM, GRU, word2vec, seq2seq without attention, traditional NLP)",
    "x3": "non-transformer CV methods (CNN, ConvNets, ResNet, VGG, pooling, convolutional architectures)",
    "x4": "multimodal methods bridging multiple modalities (combining text and images, vision-language models, image captioning, VQA)",
    "x5": "statistical machine learning (non-neural methods like SVM, random forests, logistic regression, Bayesian methods, traditional ML)",
}

# Feasibility Expansion
FEASIBILITY_PARENT_TOPICS = {
    "x1": "blockchain technology (consensus protocols, smart contracts, decentralized ledgers, proof-of-work, proof-of-stake)",
    "x2": "cryptography (encryption, zero-knowledge proofs, hash functions, digital signatures, key exchange)",
    "x3": "distributed systems (consensus algorithms, fault tolerance, replication, distributed databases, CAP theorem)",
}

FEASIBILITY_TOPICS = {
    "x1": "blockchain technology (consensus protocols, smart contracts, decentralized ledgers, proof-of-work, proof-of-stake)",
    "x2": "cryptography (encryption, zero-knowledge proofs, hash functions, digital signatures, key exchange)",
    "x3": "distributed systems (consensus algorithms, fault tolerance, replication, distributed databases, CAP theorem)",
}

# Support Expansion
SUPPORT_PARENT_TOPICS = {
    "y1": "theoretical computer science",
    "y2": "economics",
}

SUPPORT_CHILD_TOPICS = {
    "x1": "complexity theory (computational complexity, P vs NP, complexity classes, hardness results, reductions, circuit complexity, space/time complexity bounds)",
    "x2": "macroeconomics (GDP, inflation, monetary policy, fiscal policy, business cycles, economic growth, unemployment, central banking, aggregate demand/supply)",
    "x3": "mechanism design (auction design, incentive compatibility, social choice, matching markets, market design, algorithmic game theory, incentive mechanisms)",
}


# =============================================================================
# GENERIC LABEL MAPPINGS (for analysis)
# =============================================================================

OUTPUT_DIMS = {
    "binding_set": ["x1", "x2", "x3", "x4", "x5"],
    "feasibility": ["x1", "x2", "x3"],
    "support": ["x1", "x2", "x3"],
}

LIST_LABELS = {
    "binding_set": {
        "L_A": "L_y1",
        "L_B": "L_y2",
        "L_A_B_agg": "L_intersection",
        "L_f": "L_x1_or",
    },
    "feasibility": {
        "L_A": "L2",
        "L_B": "L3",
        "L_A_B_agg": "L4",
        "L_f": "L1",
    },
    "support": {
        "L_A": "L2_y1",
        "L_B": "L2_y2",
        "L_A_B_agg": "L2",
        "L_f": "L1",
    },
}


def get_list_name(experiment: str, generic_label: str) -> str:
    """Get the specific list name for a generic label in an experiment."""
    return LIST_LABELS[experiment][generic_label]


def get_vector_with_stats(data: dict, experiment: str, generic_label: str) -> dict:
    """Get full aggregate stats {dim: {mean, std, values}} for a generic label."""
    list_name = get_list_name(experiment, generic_label)
    return data["aggregates"][list_name]


def get_vector(data: dict, experiment: str, generic_label: str) -> dict:
    """Get aggregate mean vector {dim: mean} for a generic label."""
    list_name = get_list_name(experiment, generic_label)
    agg = data["aggregates"][list_name]
    return {dim: agg[dim]["mean"] for dim in agg.keys()}
