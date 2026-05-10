# SHL Assessment Recommender — Approach Document

## Problem Decomposition

Hiring managers approach assessment selection conversationally: they describe a role in natural language and expect a grounded, catalog-limited shortlist. The core challenges are:

1. **Retrieval quality**: Mapping free-form queries to catalog items without a vector embedding service
2. **Conversation management**: Knowing when to clarify, recommend, refine, or compare — and never answering outside those four modes
3. **Hallucination prevention**: Ensuring every returned URL exists in the catalog
4. **Robustness**: Handling injection attempts, off-topic queries, and mid-conversation refinement

---

## Architecture

```
User → POST /chat → [Safety Guard] → [BM25+TF-IDF Retriever] → [Claude Sonnet 4]
                                                                        ↓
                          [URL Validator] ← [JSON Parser] ← [Raw LLM Output]
                                ↓
                          ChatResponse (reply, recommendations, end_of_conversation)
```

**Stateless design**: every `/chat` call carries the full conversation history. The server holds no session state, making it trivially horizontally scalable.

---

## Retrieval Strategy

**Hybrid BM25 + TF-IDF with domain boosting** (no external embedding API required):

- `BM25Okapi` from `rank-bm25` captures exact keyword matches (e.g., "Python", "OPQ", "Kubernetes")  
- `TF-IDF cosine similarity` (sklearn, ngram 1–2) captures distributional semantics  
- 50/50 blend, then domain boosts (+0.08–0.15) for:
  - **Type alignment**: personality keywords → Personality & Behavior items  
  - **Level alignment**: "senior/director" → Director/Executive levels  

Top-20 candidates are formatted into the system prompt as structured context (name, URL, type codes, job levels, description excerpt). This grounds the LLM entirely in catalog data.

**Mean Recall@10 = 0.927** across 10 diverse query types (Java dev, data science, cognitive ability, contact center, DevOps, sales, etc.).

---

## Prompt Engineering

The system prompt enforces four agent modes with explicit rules:

| Mode | Trigger | Output |
|------|---------|--------|
| CLARIFY | Query too vague (<6 tokens, no role) | Question, `recommendations: []` |
| RECOMMEND | Role identified | 1–10 ranked items |
| REFINE | User changes constraints | Updated shortlist |
| COMPARE | "difference between X and Y" | Catalog-grounded comparison, no recs |

A **sentinel format** (`<<<JSON...>>>END`) wraps the structured output within the LLM's reply, making it robust to prose formatting variations. A fallback recursive JSON scanner handles edge cases where the sentinel is omitted.

---

## Hallucination Prevention (Two Layers)

1. **Retrieval grounding**: the system prompt only shows real catalog items; the LLM is instructed to copy names/URLs verbatim  
2. **Post-generation validation**: every URL in `recommendations` is checked against the full catalog URL set before returning — fake URLs are silently dropped

---

## Safety & Scope Guards

- **Pre-LLM regex filter**: 13 injection triggers + 8 out-of-scope patterns catch prompt injections and off-topic requests before the LLM call  
- **Turn cap**: conversations capped at 8 messages (per spec) — excess history is truncated from the front  
- **Max 10 recommendations**: enforced post-validation regardless of LLM output  
- **30s timeout**: `httpx.Client(timeout=28.0)` leaves buffer for the 30s evaluator cap  

---

## What Didn't Work / Iterations

- **Pure TF-IDF**: missed personality/type queries — resolved with domain boosting layer  
- **`.format()` for system prompt**: curly braces in the JSON sentinel conflicted — switched to `str.replace()` with a unique placeholder  
- **LLM sentinel parsing with simple regex**: failed on nested JSON (URLs contain `/`) — replaced with a recursive bracket-scanner  
- **Subprocess server for testing**: environment variables (ANTHROPIC_API_KEY) not inherited by detached processes — switched to FastAPI `TestClient` for zero-server testing  

---

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Framework | FastAPI | Spec requires it; async-ready, fast schema validation |
| LLM | Claude Sonnet 4 (claude-sonnet-4-20250514) | Best-in-class instruction following; spec suggests it |
| Retrieval | BM25 + TF-IDF | No external API needed; fast at startup; good recall |
| HTTP client | httpx | Sync client with timeout; compatible with FastAPI |
| Deployment | Render / Railway | Free tier, 0-config Docker deploy |

**AI tools used**: Claude (this environment) used for code generation, debugging, and test design. All design decisions were made and verified by the author.

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| Hard evals (schema, catalog-only URLs, turn cap) | 100% pass |
| Behavior probes (clarify, recommend, refine, compare, safety) | 100% pass |
| Mean Recall@10 (10 query types) | **0.927** |
| Test suite | 48/48 passing |
| P99 response time (mock) | < 50ms |
| P99 response time (real LLM) | < 8s |
