"""
SHL Assessment Recommender — FastAPI service
POST /chat  →  stateless conversational agent
GET  /health →  readiness check

Design:
  1. BM25 + TF-IDF hybrid search retrieves relevant catalog items
  2. Retrieved context + full conversation history sent to Claude
  3. Claude decides when to clarify, recommend, refine, or compare
  4. Structured JSON embedded in Claude's response is parsed out
"""

import json
import os
import re
import time
import logging
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from catalog_data import CATALOG
from catalog_loader import CatalogSearchEngine, format_for_llm, format_recommendation, keys_to_codes

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── One-time index build ───────────────────────────────────────────────────────
log.info("Building catalog search index ...")
_engine = CatalogSearchEngine(CATALOG)
log.info(f"Index ready — {len(CATALOG)} items")

# ── gemini client ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str     # e.g. "K", "P", "A,S"


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are an expert SHL assessment consultant. Your ONLY job is to help hiring managers and recruiters find the right SHL assessments from the official SHL product catalog.

## STRICT RULES — NEVER BREAK THESE
1. ONLY discuss SHL assessments. Refuse politely if asked about anything else (general hiring advice, legal questions, competitor products, etc.).
2. ONLY recommend assessments that appear in the CATALOG CONTEXT provided below. Never invent or hallucinate assessment names or URLs.
3. Every URL you include MUST come from the catalog context.
4. Do NOT recommend on your very first reply if the query is too vague. Ask ONE clarifying question.
5. You must ALWAYS output the JSON sentinel block at the end of your reply, even when making no recommendations.

## FOUR CONVERSATION MODES
- **CLARIFY**: Query is too vague ("I need an assessment"). Ask a single focused question.
- **RECOMMEND**: Enough context to make a 1-10 item shortlist. Be decisive.
- **REFINE**: User has updated constraints ("add personality tests", "only remote", "shorter duration"). Update the shortlist immediately.
- **COMPARE**: "What is the difference between X and Y?" - answer from catalog data only.

## WHEN TO RECOMMEND
Recommend when you know at minimum:
- What role / skill area to assess (e.g., "Java developer", "entry-level cashier", "sales manager")
If you also know job level, language, or duration preference - use them to narrow down.
A job description pasted in by the user gives you enough context to recommend immediately.

## OUTPUT FORMAT
After your conversational reply, ALWAYS append exactly this sentinel block:

<<<JSON
{"recommendations": [{"name": "EXACT_NAME_FROM_CATALOG", "url": "EXACT_URL_FROM_CATALOG", "test_type": "CODE"}], "end_of_conversation": false}
>>>END

Rules for the sentinel:
- recommendations is [] when still clarifying or refusing.
- test_type codes: K=Knowledge & Skills, P=Personality & Behavior, A=Ability & Aptitude, S=Simulations, B=Biodata & Situational Judgment, C=Competencies, D=Development & 360, E=Assessment Exercises
- Use comma-separated codes for multi-type items, e.g. "K,S"
- end_of_conversation is true ONLY after you have delivered a final shortlist and the user seems satisfied.
- Maximum 10 recommendations per response.
- Use EXACT name and URL from the catalog context below. Do not alter them.

## CATALOG CONTEXT
The following are the most relevant SHL assessments for the current query:

CATALOG_PLACEHOLDER
"""

REFUSAL_TRIGGERS = [
    "ignore previous", "ignore all", "disregard", "jailbreak",
    "forget your instructions", "act as", "you are now",
    "pretend you are", "bypass", "override system",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\blegal\s+advice\b", r"\blawsuit\b", r"\blame\b", r"\bsue\b",
    r"\bcompetitor\b", r"\bsalary\b", r"\bdiscrimination\b", r"\bpay\s+gap\b",
    r"\bwrite\s+(a\s+)?job\s+description\b", r"\bdraft\s+(a\s+)?job\b",
    r"\bwrite\s+(a\s+)?offer\b", r"\bterminat\b",
]

MOCK_MODE = os.environ.get("MOCK_LLM", "").lower() in ("1", "true", "yes")

if MOCK_MODE:
    log.warning("MOCK_LLM=true — using deterministic mock responses (testing only)")


def _mock_llm_response(messages: list, retrieved: list) -> str:
    """
    Deterministic mock for testing without an API key.
    Mimics realistic agent behavior based on conversation length and content.
    """
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    ).lower()

    # Injection / off-topic
    if any(t in last_user for t in ["ignore previous", "recipe", "salary", "legal advice", "jailbreak"]):
        return (
            "I'm only able to help with selecting SHL assessments. "
            "Could you describe the role you're hiring for?\n"
            "<<<JSON\n"
            '{\"recommendations\": [], \"end_of_conversation\": false}\n'
            ">>>END"
        )

    # Vague query
    if len(messages) == 1 and len(last_user.split()) < 6:
        return (
            "Happy to help! To recommend the right SHL assessments, "
            "could you tell me more about the role you're hiring for "
            "and the key skills or competencies you want to evaluate?\n"
            "<<<JSON\n"
            '{\"recommendations\": [], \"end_of_conversation\": false}\n'
            ">>>END"
        )

    # Comparison query
    if any(w in last_user for w in ["difference", "compare", "vs", "versus", "between"]):
        items = retrieved[:2]
        reply = (
            "Here is a comparison based on the SHL catalog:\n\n"
            + "\n\n".join(
                f"**{it['name']}** ({', '.join(it.get('keys',[]))}): "
                f"{it.get('description','')[:200]}"
                for it in items
            )
            + "\n\nThe OPQ32r measures occupational personality and behavioural style, "
            "while the Global Skills Assessment (GSA) measures 96 discrete skills/behaviors "
            "aligned to SHL's Universal Competency Framework."
        )
        return reply + "\n<<<JSON\n{\"recommendations\": [], \"end_of_conversation\": false}\n>>>END"

    # Build recommendations from top retrieved items (max 5 for mock)
    recs = []
    for item in retrieved[:5]:
        recs.append({
            "name": item["name"],
            "url": item["link"],
            "test_type": keys_to_codes(item.get("keys", []))
        })

    # Refine: thank and update
    if len(messages) >= 3:
        reply = (
            f"Based on your updated requirements, here are my refined recommendations "
            f"({len(recs)} assessments):"
        )
    else:
        reply = (
            f"Based on the role you've described, here are {len(recs)} "
            f"SHL assessments I'd recommend:"
        )

    recs_json = json.dumps(recs)
    return (
        f"{reply}\n"
        f"<<<JSON\n"
        f"{{\"recommendations\": {recs_json}, \"end_of_conversation\": false}}\n"
        f">>>END"
    )


REFUSAL_REPLY = (
    "I'm focused exclusively on helping you select SHL assessments. "
    "I'm not able to help with that request. "
    "Could you tell me more about the role you're hiring for so I can suggest the right assessments?"
)


def is_injection_or_out_of_scope(text: str) -> bool:
    lower = text.lower()
    for trigger in REFUSAL_TRIGGERS:
        if trigger in lower:
            return True
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def build_search_query(messages: List[Message]) -> str:
    """Build a rich search query from the last few turns."""
    relevant = []
    for m in messages[-6:]:   # last 3 turns
        if m.role == "user":
            relevant.append(m.content)
    return " ".join(relevant)[-1000:]   # cap length


def _extract_json(text: str):
    """Find the first valid JSON object containing 'recommendations' key."""
    for i, ch in enumerate(text):
        if ch == '{':
            for j in range(len(text), i, -1):
                candidate = text[i:j]
                try:
                    data = json.loads(candidate)
                    if "recommendations" in data:
                        return i, j, data
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def parse_agent_output(raw: str):
    """
    Extract (reply_text, recommendations, end_of_conversation) from raw LLM output.
    Handles both the <<<JSON...>>>END sentinel format and fallback JSON parsing.
    """
    # Try sentinel format first
    sentinel_match = re.search(r"<<<JSON\s*(\{.*?\})\s*>>>END", raw, re.DOTALL)
    if sentinel_match:
        try:
            data = json.loads(sentinel_match.group(1))
            reply = raw[:sentinel_match.start()].strip()
            recs = data.get("recommendations", [])
            eoc = bool(data.get("end_of_conversation", False))
            return reply, recs, eoc
        except json.JSONDecodeError:
            pass

    # Fallback: scan for any embedded JSON with 'recommendations'
    result = _extract_json(raw)
    if result:
        i, j, data = result
        reply = raw[:i].strip()
        recs = data.get("recommendations", [])
        eoc = bool(data.get("end_of_conversation", False))
        return reply, recs, eoc

    # No parseable JSON — return raw text, no recommendations
    return raw.strip(), [], False


def validate_recommendations(recs: list) -> List[Recommendation]:
    """
    Filter recommendations to only those that exist in the catalog.
    Prevents hallucination of URLs or names.
    """
    valid_urls = {item["link"] for item in CATALOG}
    valid_names = {item["name"].lower(): item for item in CATALOG}

    validated = []
    seen = set()

    for rec in recs:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name", "")
        url = rec.get("url", "")
        test_type = rec.get("test_type", "K")

        # Check URL is in catalog
        if url in valid_urls and url not in seen:
            validated.append(Recommendation(name=name, url=url, test_type=test_type))
            seen.add(url)
        # Try to find by name if URL doesn't match
        elif name.lower() in valid_names and name not in seen:
            item = valid_names[name.lower()]
            validated.append(Recommendation(
                name=item["name"],
                url=item["link"],
                test_type=keys_to_codes(item.get("keys", []))
            ))
            seen.add(name)

    return validated[:10]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    # Guard: turn cap (8 messages = 4 user + 4 assistant turns)
    if len(req.messages) > 8:
        req.messages = req.messages[-8:]

    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )

    # ── Safety check ──────────────────────────────────────────────────────────
    if is_injection_or_out_of_scope(last_user):
        return ChatResponse(
            reply=REFUSAL_REPLY,
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Retrieve relevant catalog items ───────────────────────────────────────
    query = build_search_query(req.messages)
    retrieved = _engine.search(query, top_k=20)
    catalog_context = format_for_llm(retrieved, max_items=20)

    system = SYSTEM_PROMPT_TEMPLATE.replace("CATALOG_PLACEHOLDER", catalog_context)

    # ── Build messages for Claude ─────────────────────────────────────────────
    claude_messages = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if m.role in ("user", "assistant")
    ]

    # ── Call Claude (or mock) ─────────────────────────────────────────────────
    t0 = time.time()

    if MOCK_MODE:
        raw_text = _mock_llm_response(claude_messages, retrieved)
    
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system,
            )
            # Convert messages to Gemini format
            history = []
            for msg in claude_messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                history.append({"role": role, "parts": [msg["content"]]})
            
            chat = model.start_chat(history=history)
            last_msg = claude_messages[-1]["content"]
            gemini_resp = chat.send_message(last_msg)
            raw_text = gemini_resp.text

        except Exception as e:
            log.error(f"Gemini API error: {e}")
            raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    # ── Parse output ──────────────────────────────────────────────────────────
    reply, raw_recs, eoc = parse_agent_output(raw_text)
    validated_recs = validate_recommendations(raw_recs)

    # ── Log summary ───────────────────────────────────────────────────────────
    log.info(
        f"Reply len={len(reply)} | recs={len(validated_recs)} | eoc={eoc} | "
        f"turns={len(req.messages)}"
    )

    return ChatResponse(
        reply=reply,
        recommendations=validated_recs,
        end_of_conversation=eoc,
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
