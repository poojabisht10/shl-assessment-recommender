# SHL Assessment Recommender

> A conversational AI agent that recommends the right SHL assessments from the official product catalog — through natural dialogue, not keyword search.

**Live API:** https://shl-assessment-recommender-mj5i.onrender.com

---

## What It Does

Hiring managers describe a role in plain English. The agent asks clarifying questions, retrieves relevant assessments from the 352-item SHL catalog, and returns a grounded shortlist — with names, URLs, and test type codes. It handles four conversation modes:

| Mode          | Example                           | Behaviour                 |
| ------------- | --------------------------------- | ------------------------- |
| **Clarify**   | "I need an assessment"            | Asks one focused question |
| **Recommend** | "Hiring a mid-level Java dev"     | Returns 1–10 assessments  |
| **Refine**    | "Add a personality test too"      | Updates the shortlist     |
| **Compare**   | "Difference between OPQ and GSA?" | Catalog-grounded answer   |

---

## Live Demo

```bash
# Health check
curl https://shl-assessment-recommender-mj5i.onrender.com/health

# Get recommendations
curl -X POST https://shl-assessment-recommender-mj5i.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a mid-level Java developer with 4 years experience"},
      {"role": "assistant", "content": "Are you looking for core Java skills or specific frameworks?"},
      {"role": "user", "content": "Core Java and Java 8, they also work with business stakeholders"}
    ]
  }'
```

---

## API Reference

### `GET /health`

Returns service status.

```json
{ "status": "ok" }
```

---

### `POST /chat`

Stateless endpoint — send the full conversation history on every call.

**Request body:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer who works with stakeholders"
    },
    {
      "role": "assistant",
      "content": "What seniority level are you hiring for?"
    },
    { "role": "user", "content": "Mid-level, around 4 years experience" }
  ]
}
```

**Response:**

```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java developer with stakeholder needs.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**test_type codes:**

| Code | Meaning                        |
| ---- | ------------------------------ |
| `K`  | Knowledge & Skills             |
| `P`  | Personality & Behavior         |
| `A`  | Ability & Aptitude             |
| `S`  | Simulations                    |
| `B`  | Biodata & Situational Judgment |
| `C`  | Competencies                   |
| `D`  | Development & 360              |
| `E`  | Assessment Exercises           |

---

## Run Locally

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) (takes 2 mins)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/shl-assessment-recommender.git
cd shl-assessment-recommender

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY=gsk_your-key-here

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server runs at: `http://localhost:8000`

---

## Run Tests

```bash
# Full test suite with mock LLM (no API key needed) — 48 tests
MOCK_LLM=true python3 run_tests.py

# Integration tests against live deployed URL
python3 test_api.py https://shl-assessment-recommender-mj5i.onrender.com
```

**Test results:**

| Metric                                                        | Result       |
| ------------------------------------------------------------- | ------------ |
| Hard evals (schema, catalog-only URLs, turn cap)              | 100%         |
| Behavior probes (clarify, recommend, refine, compare, safety) | 100%         |
| Mean Recall@10 across 10 query types                          | 0.927        |
| Total tests                                                   | 48/48 passed |

---

## Deploy

### Render (Free, Recommended)

```
1. Push to GitHub
2. render.com → New Web Service → Connect repo
3. Build command:  pip install -r requirements.txt
4. Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
5. Add env var:    GROQ_API_KEY = gsk_your-key-here
6. Deploy
```

### Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your-key-here shl-recommender
```

---

## Environment Variables

| Variable       | Required | Description                                  |
| -------------- | -------- | -------------------------------------------- |
| `GROQ_API_KEY` | Yes      | Groq API key — get free at console.groq.com  |
| `MOCK_LLM`     | No       | Set to `true` for testing without an API key |

---

## Architecture

```
POST /chat
    │
    ├── Safety Guard (regex — blocks injections, off-topic)
    │
    ├── Hybrid Retriever (BM25 + TF-IDF + domain boosting)
    │       └── Top 20 catalog items → injected into system prompt
    │
    ├── Groq LLM (llama-3.3-70b-versatile)
    │       └── Decides: clarify / recommend / refine / compare
    │
    └── URL Validator (drops any URL not in the 352-item catalog)
            └── ChatResponse (reply, recommendations, end_of_conversation)
```

**Key design decisions:**

- **Stateless** — full conversation history sent on every request; no session storage needed
- **No vector DB** — BM25 + TF-IDF runs entirely in-memory at startup in under 100ms
- **Two-layer hallucination guard** — LLM grounded by retrieved context; post-generation URL validation drops anything not in the catalog
- **Sentinel output format** (`<<<JSON...>>>END`) with recursive fallback parser — robust to any LLM formatting variation
- **Max 8 turns, max 10 recommendations, 28s timeout** — all enforced server-side per spec

---

## Project Structure

```
shl-assessment-recommender/
├── main.py              # FastAPI app — /health and /chat endpoints
├── catalog_loader.py    # Hybrid BM25+TF-IDF search engine
├── catalog_data.py      # 352 SHL catalog items (cleaned)
├── run_tests.py         # 48-test suite (FastAPI TestClient)
├── test_api.py          # Integration tests for live URL
├── requirements.txt     # Pinned dependencies
├── Dockerfile           # Container build
├── render.yaml          # One-click Render deployment
└── APPROACH.md          # Design document
```

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [Groq](https://groq.com/) — LLM inference (llama-3.3-70b-versatile)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 retrieval
- [scikit-learn](https://scikit-learn.org/) — TF-IDF retrieval
- [Render](https://render.com/) — Deployment
