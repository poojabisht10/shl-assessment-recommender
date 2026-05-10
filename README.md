# SHL Assessment Recommender API

Conversational agent that recommends SHL assessments from the official product catalog.

## Quick Start (Local)

```bash
# Clone / download the files, then:
pip install -r requirements.txt

# With real Claude API:
ANTHROPIC_API_KEY=sk-ant-... uvicorn main:app --host 0.0.0.0 --port 8000

# With mock LLM (no API key needed, for testing):
MOCK_LLM=true uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What seniority level?"},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java developer with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)",              "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**test_type codes:** `K`=Knowledge, `P`=Personality, `A`=Ability, `S`=Simulations, `B`=Biodata, `C`=Competencies, `D`=Development, `E`=Exercises

## Deploy to Render (Free)

1. Push this folder to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set environment variable: `ANTHROPIC_API_KEY = sk-ant-...`
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Deploy — your URL will be `https://your-app.onrender.com`

## Deploy via Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

## Run Tests

```bash
# Unit + integration tests (no API key needed):
MOCK_LLM=true python3 run_tests.py

# Against a live deployed endpoint:
python3 test_api.py https://your-app.onrender.com
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (production) | Anthropic API key |
| `MOCK_LLM` | No | Set to `true` to use mock responses (testing) |

## Design Notes

- **Stateless**: every `/chat` call sends the full conversation history
- **Hybrid retrieval**: BM25 + TF-IDF + domain boosting (no vector DB needed)
- **Hallucination guard**: all returned URLs are validated against the 352-item catalog
- **Safety**: pre-LLM regex filter blocks injections and out-of-scope requests
- **Limits**: max 8 turns, max 10 recommendations, 28s LLM timeout
