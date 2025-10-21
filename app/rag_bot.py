from fastapi import FastAPI
import os
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from difflib import SequenceMatcher

app = FastAPI()

INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_index_metadata.pkl"
VECTORSTORE_PATH = "vectorstore.index"

import json
from pathlib import Path

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")


def load_documents():
    pdf_files = [
        "./datasets/hiv_information_sheets.pdf",
        "./datasets/hiv_qa.pdf"
    ]

    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        raw_docs = loader.load()

        # ✅ Add metadata to each page/document
        for doc in raw_docs:
            doc.metadata["source"] = pdf

        chunks = splitter.split_documents(raw_docs)
        documents.extend(chunks)

    return documents


def get_vectorstore():
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=api_key) # type: ignore

    if os.path.exists(VECTORSTORE_PATH):
        print("Loading cached vectorstore...")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vectorstore and saving cache...")
        documents = load_documents()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore


def get_chatbot():
    vectorstore = get_vectorstore()
    chat = ChatCohere(model="command-a-03-2025", temperature=0, cohere_api_key=api_key) # type: ignore
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# ✅ Stronger spacing and OCR artifact cleaner
def fix_spacing(text: str) -> str:
    # Remove excessive spaces between single letters like "H I V" → "HIV"
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b", r"\1\2\3", text)
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\b", r"\1\2", text)

    # Add missing spaces before capital letters (e.g. "HowdoesHIVattack" → "How does HIV attack")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # Add spaces before known words (HIV, AIDS, PrEP, ART, STI, infection, etc.)
    keywords = ["HIV", "AIDS", "PrEP", "ART", "STI", "infection", "body", "transmitted", "attack", "signs", "same", "come", "from"]
    for word in keywords:
        text = re.sub(rf"(?i)(?<!\s)({word})", r" \1", text)

    # Fix glued lowercase/uppercase transitions: "WhatisHIV" → "What is HIV"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure proper capitalization and question mark spacing
    text = text[0].upper() + text[1:] if text else text
    text = re.sub(r"\s+\?", "?", text)

    return text

# ✅ Generate 5 clean suggested questions
def generate_suggested_questions(query: str, answer: str) -> list[str]:
    """Generate 5 clean follow-up questions from HIV PDFs."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 15, "score_threshold": 0.3}
    )

    docs = retriever.get_relevant_documents(query + " " + answer)
    docs = [
        d for d in docs
        if any(x in d.metadata.get("source", "").lower()
               for x in ["hiv_qa.pdf", "hiv_information_sheets.pdf"])
    ]

    context = "\n".join([doc.page_content for doc in docs])

    potential_questions = re.findall(r"([A-Z][^?.!]{3,120}\?)", context)
    cleaned, seen = [], set()

    for q in potential_questions:
        q = re.sub(r"(?i)\bTopic\s*\d+[:.\-]?\s*", "", q)
        q = re.sub(r"(?i)\bQ\d+[:.\-]?\s*", "", q)
        q = re.sub(r"PMC\d+/?", "", q)
        q = re.sub(r"HIVChatbot Dataset", "", q, flags=re.I)
        q = fix_spacing(q)

        if not q.endswith("?"):
            q += "?"

        if 10 < len(q) < 120 and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q)

    # Sort by similarity
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    ranked = sorted(
        cleaned,
        key=lambda x: max(similarity(x, query), similarity(x, answer)),
        reverse=True
    )

    # Remove near-duplicates
    unique_ranked = []
    for q in ranked:
        if not any(SequenceMatcher(None, q.lower(), r.lower()).ratio() > 0.9 for r in unique_ranked):
            unique_ranked.append(q)

    fallback = [
        "What are the common symptoms of HIV?",
        "How can HIV be prevented?",
        "Can HIV be transmitted through casual contact?",
        "How is HIV treated?",
        "Where can I get tested for HIV?"
    ]

    return unique_ranked[:5] or fallback
