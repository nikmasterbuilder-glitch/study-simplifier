from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)  # logs INFO and above
logger = logging.getLogger(__name__)

print("API key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

app = FastAPI()

app = FastAPI()


@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log full traceback
        logger.error(f"Error on request {request.method} {request.url}")
        logger.error(traceback.format_exc())
        # Optional: return a friendly JSON response instead of raw 500
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


genai.configure(api_key="GOOGLE_API_KEY")
model = genai.GenerativeModel("gemini-2.5-flash")


class URLRequest(BaseModel):
    url: str


PROMPT_TEMPLATE = """
You are rewriting a scientific abstract for someone with no medical or scientific background.

Rules:
- Use only information explicitly stated in the abstract.
- Do not add background knowledge or explanations.
- Do not give medical advice or recommendations.
- Do not judge whether the treatment or supplement works.
- Do not use technical jargon; replace it with simple everyday language.

Format:
- Maximum of 8 bullet points
- Each bullet should be 1 sentence
- Neutral, descriptive tone

Focus only on:
- What was studied
- How the study was conducted
- What was observed or measured
- The study’s stated conclusion

Abstract:
{abstract}
"""


@app.post("/summarize")
def summarize_study(data: URLRequest):
    url = data.url

    # Fetch page
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    abstract_div = soup.find("div", class_="abstract-content")

    if not abstract_div:
        return {"error": "Abstract not found"}

    abstract_text = abstract_div.get_text(strip=True)

    prompt = PROMPT_TEMPLATE.format(abstract=abstract_text)

    ai_response = model.generate_content(prompt)

    return {
        "summary": ai_response.text,
        "original_abstract": abstract_text
    }
