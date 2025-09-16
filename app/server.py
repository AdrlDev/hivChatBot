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
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split into lines for easier handling
    lines = text.split("\n")

    formatted_lines = []
    for line in lines:
        line = line.strip()

        # Handle numbered lists (e.g., "1. Something")
        if re.match(r"^\d+\.", line):
            formatted_lines.append(line)

        # Handle unordered lists (markdown-style or plain dash/asterisk)
        elif re.match(r"^[-*•]\s+", line):
            item = re.sub(r"^[-*•]\s*", "", line).strip()
            formatted_lines.append(f"• {item}")

        # Normal sentence
        else:
            # Ensure first letter is capitalized
            if line and not line[0].isupper():
                line = line[0].upper() + line[1:]
            formatted_lines.append(line)

    # Join lines with proper spacing
    formatted_text = "\n".join(formatted_lines)

    # Add justification (ensure periods at end of sentences)
    formatted_text = re.sub(r"(?<![.!?])\s*$", ".", formatted_text)

    return formatted_text


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