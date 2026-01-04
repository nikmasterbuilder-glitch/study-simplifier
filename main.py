from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from google import genai

app = FastAPI()

client = genai.Client(api_key="AIzaSyBO_mPsKI-wa6Kpj-jkKAHsBZrLuQj7VWM")


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

    ai_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    )

    return {
        "summary": ai_response.text,
        "original_abstract": abstract_text
    }
