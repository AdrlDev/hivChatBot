#server.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import FastAPI
from .rag_bot import get_chatbot  # <-- your actual rag module
import re

app = FastAPI()
qa_bot = get_chatbot()  # Re-enable this to use your RAG model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_response(text: str) -> str:
    # Remove Markdown symbols and list markers
    text = re.sub(r"[*_`#>-]+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Ensure common list patterns become proper sentences
    text = re.sub(r"(\d+\.)\s*", r"\1 ", text)   # "1. " stays clean
    text = re.sub(r"\s-\s", ". ", text)          # "- Symptom" -> ". Symptom"

    # Fix missing periods between items (if not already there)
    text = re.sub(r"(?<=[a-zA-Z])\s(?=[A-Z])", ". ", text)

    # Trim leading/trailing spaces
    text = text.strip()

    # Ensure it ends with a period
    if not text.endswith("."):
        text += "."

    return text

@app.get("/chat")
def chat(query: str = Query(...)):
    try:
        result = qa_bot.invoke(query)
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        clean_answer = clean_response(answer)

        if not sources:
            return {
                "answer": {
                    "query": query,
                    "result": "I'm sorry, I couldn't find an exact answer, but I can try to help further if you rephrase your question."
                }
            }

        return {
            "answer": {
                "query": query,
                "result": clean_answer
            },
            "sources": [doc.metadata for doc in sources]  # optional: show file/line
        }

    except Exception as e:
        return {
        "answer": {
            "query": query,
            "result": "Something went wrong while processing your request."
            },
            "error": str(e)  # optional, for debugging
        }