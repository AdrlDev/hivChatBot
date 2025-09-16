#server.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import FastAPI
from .rag_bot import get_chatbot, generate_suggested_questions  # <-- your actual rag module
import re

app = FastAPI()
qa_bot = get_chatbot()  # Re-enable this to use your RAG model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_response(text: str) -> str:
    # Remove markdown symbols
    text = re.sub(r"[*_`#>-]+", "", text)

    # Replace multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # ✅ Ensure each numbered list item starts on a new line
    text = re.sub(r"(\d+)\.\s*", r"\n\1. ", text)

    # ✅ Ensure bullet points also start on new lines
    text = re.sub(r"(•|-)\s*", r"\n• ", text)

    # ✅ Capitalize first character of sentences if missing
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line[0].isupper():
            line = line[0].upper() + line[1:]
        sentences.append(line)

    formatted_text = "\n".join(sentences)

    return formatted_text.strip()


@app.get("/chat")
def chat(query: str = Query(...)):
    try:
        result = qa_bot.invoke(query)
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        clean_answer = format_response(answer)

        # ✅ Generate 5 suggested questions
        suggestions = generate_suggested_questions(query, clean_answer)

        if not sources:
            return {
                "answer": {
                    "query": query,
                    "result": "I'm sorry, I couldn't find an exact answer, but I can try to help further if you rephrase your question."
                },
                "suggested_questions": suggestions
            }

        return {
            "answer": {
                "query": query,
                "result": clean_answer
            },
            "sources": [doc.metadata for doc in sources],
            "suggested_questions": suggestions
        }

    except Exception as e:
        return {
        "answer": {
            "query": query,
            "result": "Something went wrong while processing your request."
            },
            "error": str(e)  # optional, for debugging
        }