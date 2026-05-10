"""
Catalog loading, indexing, and search for SHL Assessment Recommender.
Uses BM25 + TF-IDF cosine similarity hybrid for robust retrieval.
"""

import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Test type key mapping ────────────────────────────────────────────────────
KEY_CODE_MAP = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

def keys_to_codes(keys: List[str]) -> str:
    codes = []
    for k in keys:
        c = KEY_CODE_MAP.get(k)
        if c and c not in codes:
            codes.append(c)
    return ",".join(codes) if codes else "K"


def build_document(item: Dict) -> str:
    """Build a rich text document for each catalog item for indexing."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
        item.get("duration", ""),
    ]
    return " ".join(p for p in parts if p)


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


class CatalogSearchEngine:
    def __init__(self, catalog: List[Dict]):
        self.catalog = catalog
        self._build_index()

    def _build_index(self):
        docs = [build_document(item) for item in self.catalog]
        tokenized = [tokenize(d) for d in docs]

        # BM25
        self.bm25 = BM25Okapi(tokenized)

        # TF-IDF
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
        self.tfidf_matrix = self.tfidf.fit_transform(docs)

        self.docs = docs

    def _type_boost(self, query: str, item: Dict) -> float:
        """Extra score when query intent matches assessment type."""
        q = query.lower()
        keys = [k.lower() for k in item.get("keys", [])]
        boost = 0.0

        # Personality/behavior keywords
        if any(w in q for w in ["personality", "behavior", "behaviour", "trait", "opq", "motivation",
                                  "leadership", "team", "emotional", "sales", "mqm"]):
            if any("personality" in k or "behavior" in k for k in keys):
                boost += 0.15

        # Ability/cognitive keywords
        if any(w in q for w in ["cognitive", "ability", "aptitude", "numerical", "verbal",
                                  "deductive", "inductive", "reasoning", "iq", "g+"]):
            if any("ability" in k or "aptitude" in k for k in keys):
                boost += 0.15

        # Simulation keywords
        if any(w in q for w in ["simulation", "coding", "practical", "hands-on", "real-world",
                                  "automata", "typing", "data entry"]):
            if any("simulation" in k for k in keys):
                boost += 0.10

        # Competency/360 keywords
        if any(w in q for w in ["competency", "360", "feedback", "development", "hipo", "potential"]):
            if any("competenc" in k or "development" in k for k in keys):
                boost += 0.10

        return boost

    def _level_boost(self, query: str, item: Dict) -> float:
        """Boost items that match seniority signals in query."""
        q = query.lower()
        levels = [l.lower() for l in item.get("job_levels", [])]
        if not levels:
            return 0.0
        boost = 0.0

        level_map = {
            ("entry", "junior", "graduate", "fresher", "intern"): ["entry-level", "graduate"],
            ("mid", "intermediate", "3", "4", "5 year"): ["mid-professional", "professional individual contributor"],
            ("senior", "principal", "lead", "staff"): ["professional individual contributor", "mid-professional"],
            ("manager", "management", "team lead"): ["manager", "front line manager", "supervisor"],
            ("director", "vp", "head of", "executive", "c-suite", "cto", "ceo"): ["director", "executive"],
        }
        for keywords, target_levels in level_map.items():
            if any(k in q for k in keywords):
                if any(t in levels for t in target_levels):
                    boost += 0.08
                    break
        return boost

    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        """Hybrid BM25 + TF-IDF + type/level boosting search."""
        n = len(self.catalog)

        # BM25 scores (normalized to [0,1])
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))
        bm25_max = bm25_scores.max()
        bm25_norm = bm25_scores / (bm25_max + 1e-9)

        # TF-IDF cosine scores
        qv = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(qv, self.tfidf_matrix)[0]

        # Base hybrid score
        hybrid = 0.5 * bm25_norm + 0.5 * tfidf_scores

        # Add type and level boosts
        for idx, item in enumerate(self.catalog):
            hybrid[idx] += self._type_boost(query, item)
            hybrid[idx] += self._level_boost(query, item)

        top_indices = np.argsort(hybrid)[::-1][:top_k]
        results = []
        for i in top_indices:
            if hybrid[i] > 0.001:
                item = dict(self.catalog[i])
                item["_score"] = float(hybrid[i])
                results.append(item)
        return results

    def get_by_name(self, name: str) -> List[Dict]:
        """Find items by exact or fuzzy name match."""
        name_lower = name.lower()
        return [
            item for item in self.catalog
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower
        ]


def format_for_llm(items: List[Dict], max_items: int = 15) -> str:
    """Format catalog items for inclusion in LLM context."""
    lines = []
    for item in items[:max_items]:
        langs = item.get("languages", [])
        lang_str = ", ".join(langs[:4]) + (f" (+{len(langs)-4} more)" if len(langs) > 4 else "")
        lines.append(
            f"- **{item['name']}** (entity_id={item['entity_id']})\n"
            f"  URL: {item['link']}\n"
            f"  Type: {keys_to_codes(item.get('keys', []))} | Keys: {', '.join(item.get('keys', []))}\n"
            f"  Job Levels: {', '.join(item.get('job_levels', [])) or 'All levels'}\n"
            f"  Duration: {item.get('duration', 'Variable') or 'Variable'}\n"
            f"  Languages: {lang_str or 'See catalog'}\n"
            f"  Description: {item.get('description', '')[:300]}"
        )
    return "\n\n".join(lines)


def format_recommendation(item: Dict) -> Dict:
    """Format a catalog item as an API recommendation object."""
    return {
        "name": item["name"],
        "url": item["link"],
        "test_type": keys_to_codes(item.get("keys", [])),
    }
