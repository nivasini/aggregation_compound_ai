"""Utilities for paper list generation, classification, and set operations."""

import os
import re
import time
from typing import List, Dict, Optional, Any, Tuple
from openai import OpenAI, RateLimitError

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
RATE_LIMIT_WAIT = 5
MAX_RETRIES = 3


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """Get OpenAI client with API key from parameter or environment."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
    return OpenAI(api_key=key)


def _call_with_retry(client, model, messages, temperature, seed):
    """Make an API call with retry on rate limit."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            return response
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RATE_LIMIT_WAIT * (attempt + 1))


# =============================================================================
# PAPER LIST GENERATION
# =============================================================================

def get_papers(
    prompt: str,
    client: OpenAI,
    seed: int,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> List[str]:
    """Get list of papers from a prompt."""
    response = _call_with_retry(
        client, model, [{"role": "user", "content": prompt}], temperature, seed
    )

    text = response.choices[0].message.content
    lines = text.strip().split('\n')
    papers = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^[\d]+[\.\)\-\s]*', '', line).strip()
        cleaned = cleaned.strip('"\'')
        if cleaned and len(cleaned) > 10:
            papers.append(cleaned)
    return papers


# =============================================================================
# TITLE NORMALIZATION & SET OPERATIONS
# =============================================================================

def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching."""
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def find_intersection(list1: List[str], list2: List[str]) -> List[str]:
    """Find papers appearing in both lists using fuzzy matching."""
    norm1 = {normalize_title(p): p for p in list1}
    norm2 = {normalize_title(p): p for p in list2}
    intersection = []
    for n1, orig1 in norm1.items():
        for n2 in norm2.keys():
            if n1 == n2 or (len(n1) > 20 and len(n2) > 20 and (n1 in n2 or n2 in n1)):
                intersection.append(orig1)
                break
    return list(set(intersection))


def find_union(list1: List[str], list2: List[str]) -> List[str]:
    """Find union of two paper lists, removing duplicates using fuzzy matching."""
    seen_normalized: Dict[str, str] = {}
    union = []

    for paper in list1:
        norm = normalize_title(paper)
        if norm not in seen_normalized:
            seen_normalized[norm] = paper
            union.append(paper)

    for paper in list2:
        norm = normalize_title(paper)
        is_duplicate = False
        if norm in seen_normalized:
            is_duplicate = True
        else:
            for existing_norm in seen_normalized.keys():
                if len(norm) > 20 and len(existing_norm) > 20:
                    if norm in existing_norm or existing_norm in norm:
                        is_duplicate = True
                        break
        if not is_duplicate:
            seen_normalized[norm] = paper
            union.append(paper)

    return union


# =============================================================================
# BATCH CLASSIFICATION
# =============================================================================

def classify_paper_list_batch(
    papers: List[str],
    topics: Dict[str, str],
    client: OpenAI,
    seed: int = 1,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, float]:
    """Classify an entire list of papers in a single API call.
    Returns the topic vector (fraction of papers in each topic)."""
    if not papers:
        return {key: 0.0 for key in topics}

    paper_list = "\n".join(f"{i+1}. {paper}" for i, paper in enumerate(papers))
    topic_list = "\n".join(f"- {key}: {desc}" for key, desc in topics.items())
    topic_keys_str = ", ".join(topics.keys())

    prompt = f"""For each of the following research papers, determine whether it belongs to each topic.
Treat each paper independently - do not let other papers in the list influence your classification.

Papers:
{paper_list}

Topics:
{topic_list}

For each paper, independently answer yes/no for each topic.

Output format:
1. {topic_keys_str.replace(', ', ': yes/no, ')}: yes/no
2. {topic_keys_str.replace(', ', ': yes/no, ')}: yes/no
...

SUMMARY (count of "yes" for each topic):
{chr(10).join(f'{key}: [count]' for key in topics.keys())}
"""

    response = _call_with_retry(
        client, model, [{"role": "user", "content": prompt}], temperature, seed
    )

    answer_text = response.choices[0].message.content.strip()
    results = {key: 0.0 for key in topics}
    n = len(papers)

    summary_section = answer_text.lower()
    if 'summary' in summary_section:
        summary_section = summary_section.split('summary', 1)[1]

    for key in topics.keys():
        pattern = rf'{key}:\s*(\d+)'
        match = re.search(pattern, summary_section)
        if match:
            count = int(match.group(1))
            if count > n:
                count = n
            results[key] = count / n if n > 0 else 0.0

    return results


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_inclusion_exclusion_prompt(
    inclusion_topics: List[str],
    exclusion_topics: List[str],
    inclusion_op: bool = True,
    exclusion_op: bool = False,
    n: int = 20,
) -> str:
    """Generate a prompt for listing papers with inclusion/exclusion criteria."""
    if inclusion_topics:
        if len(inclusion_topics) == 1:
            inc_str = f"*{inclusion_topics[0]}*"
        else:
            op = " AND " if inclusion_op else " OR "
            inc_str = op.join(f"*{t}*" for t in inclusion_topics)
        prompt = f"List up to {n} research papers on {inc_str}"
    else:
        prompt = f"List up to {n} research papers"

    if exclusion_topics:
        if len(exclusion_topics) == 1:
            prompt += f", but EXCLUDE any papers about *{exclusion_topics[0]}*."
        else:
            prompt += ", but EXCLUDE any papers about:"
            for topic in exclusion_topics:
                prompt += f"\n- {topic}"
    else:
        prompt += "."

    prompt += "\n\nOutput only paper titles, one per line."
    return prompt
