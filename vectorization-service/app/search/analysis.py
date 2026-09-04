"""Shared OpenSearch analyzer — defined once because two services must produce identical tokens.

What counts as a match:
  * `html_strip` — content is stored as HTML.
  * `separator_to_space` — runs of `- _ / . ~ +` become spaces, so "ATLAS-6" or
    "metrics-wire-dlq-depth.csv" tokenize into words the same way in a short query and inside a
    sentence.
  * English stopwords + Porter stemming.

Queries and documents must go through the same analyzer; everything else is tunable.
"""
from __future__ import annotations

from typing import Any, Dict

# Char filters run before tokenization, in order: strip markup first, then flatten separators, so
# `<a href="x/y">` does not leave `x y` behind as searchable content.
ANALYSIS_SETTINGS: Dict[str, Any] = {
    "char_filter": {
        "separator_to_space": {
            "type": "pattern_replace",
            "pattern": r"[-_/.~+]+",
            "replacement": " ",
        }
    },
    "filter": {
        "english_stop": {"type": "stop", "stopwords": "_english_"},
        "english_stemmer": {"type": "stemmer", "language": "english"},
    },
    "analyzer": {
        "orchestra_text": {
            "type": "custom",
            "char_filter": ["html_strip", "separator_to_space"],
            "tokenizer": "standard",
            "filter": ["lowercase", "english_stop", "english_stemmer"],
        },
        # Same pipeline minus stemming, for fields where the surface form is the point (issue keys,
        # filenames). Stemming "ATLAS" or "metrics" here only loses precision.
        "orchestra_identifier": {
            "type": "custom",
            "char_filter": ["html_strip", "separator_to_space"],
            "tokenizer": "standard",
            "filter": ["lowercase"],
        },
    },
}

# Single-node dev/demo cluster: a replica can never be allocated, so asking for one leaves every
# index permanently yellow and makes `wait_for_status=green` health checks hang. Set explicitly
# rather than relying on the default (1).
INDEX_SETTINGS: Dict[str, Any] = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": ANALYSIS_SETTINGS,
}
