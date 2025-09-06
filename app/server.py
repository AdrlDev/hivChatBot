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
    # Remove Markdown bold/italics
    text = re.sub(r"[*_`]+", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Summarize: keep first 2–3 sentences only
    sentences = text.split(". ")
    summary = ". ".join(sentences[:3]).strip()

    # Ensure it ends nicely
    if not summary.endswith("."):
        summary += "."

    return summary

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