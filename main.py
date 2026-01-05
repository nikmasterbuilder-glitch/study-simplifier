from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import os
import logging
import traceback

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- ENV VAR --------------------
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY is not set")

HF_MODEL = "sshleifer/distilbart-cnn-12-6"

# -------------------- APP --------------------
app = FastAPI()

# -------------------- MIDDLEWARE --------------------


@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception:
        logger.error(f"Error on request {request.method} {request.url}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error. Check logs for details."}
        )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# -------------------- MODELS --------------------


class URLRequest(BaseModel):
    url: str


# -------------------- PROMPT --------------------
PROMPT_TEMPLATE = """
Rewrite the following scientific abstract for someone with no medical or scientific background.

Rules:
- Use only information explicitly stated in the abstract.
- Do not add background knowledge.
- Do not give medical advice.
- Do not judge effectiveness.
- Replace technical language with simple everyday wording.

Format:
- Max 8 bullet points
- Each bullet one sentence
- Neutral, descriptive tone

Abstract:
{abstract}
"""

# -------------------- HF CALL --------------------


def summarize_with_hf(prompt: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "min_length": 80,
            "do_sample": False
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    return response.json()[0]["summary_text"]

# -------------------- ROUTE --------------------


@app.post("/summarize")
def summarize_study(data: URLRequest):
    # Fetch page
    response = requests.get(data.url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    abstract_div = soup.find("div", class_="abstract-content")

    if not abstract_div:
        return {"error": "Abstract not found"}

    abstract_text = abstract_div.get_text(strip=True)

    summary = summarize_with_hf(abstract_text)

    return {
        "summary": summary,
        "original_abstract": abstract_text
    }
