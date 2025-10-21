#server.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import FastAPI
from .rag_bot import get_chatbot, generate_suggested_questions  # <-- your actual rag module
import re
import random

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

# ✅ Predefined responses based on image
def get_predefined_response(query: str) -> str | None:
    q = query.lower().strip()

    # Define patterns that only match whole words
    greetings = [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood morning\b", r"\bgood afternoon\b", r"\bgood evening\b"]
    farewells = [r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\btake care\b", r"\bthank you\b", r"\bthanks\b"]
    hiv_keywords = [r"\bhiv\b", r"\baids\b", r"\bcondom\b", r"\bprep\b", r"\bart\b", r"\bsti\b", r"\binfection\b"]

    # Predefined responses
    greeting_responses = [
        "Hi there! Welcome to **HIVocate**. I'm here to assist you with HIV information you need.\n",
        "Hello! I'm your HIV information assistant. How can I help you today?\n",
        "Welcome to **HIVocate!** I'm here to provide you with accurate HIV-related information.\n"
    ]

    farewell_responses = [
        "Take care! Remember, knowledge is power when it comes to HIV prevention.\n",
        "Goodbye! Stay informed and stay safe.\n",
        "Thank you for using **HIVocate.** Have a great day!\n"
    ]

    unrelated_responses = [
        "I'm specialized to answer questions about HIV only. Please ask me about HIV.\n",
        "Outside of my knowledge area.\n"
    ]

    # 1️⃣ Greeting (use regex whole-word search)
    if any(re.search(pattern, q) for pattern in greetings):
        return random.choice(greeting_responses)

    # 2️⃣ Farewell
    if any(re.search(pattern, q) for pattern in farewells):
        return random.choice(farewell_responses)

    # 3️⃣ Not related to HIV
    if not any(re.search(pattern, q) for pattern in hiv_keywords):
        return random.choice(unrelated_responses)

    return None

@app.get("/chat")
def chat(query: str = Query(...)):
    try:
        # ✅ Step 1: Check predefined rules first
        predefined = get_predefined_response(query)

        result = qa_bot.invoke(query)
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        clean_answer = format_response(answer)

        # ✅ Generate 5 suggested questions
        suggestions = generate_suggested_questions(query, clean_answer)

        if not sources:
            if predefined:
                return {
                    "answer": {
                        "query": query,
                        "result": predefined
                    },
                    "suggested_questions": []
                }
            
                return {
                    "answer": {
                        "query": query,
                        "result": (
                            "I'm sorry, I couldn't find an exact answer to your question right now. "
                            "Please try rephrasing it or ask something specific about HIV prevention, testing, or treatment."
                        )
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