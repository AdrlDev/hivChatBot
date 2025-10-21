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

# ✅ Enhanced spacing and OCR artifact cleaner
def fix_spacing(text: str) -> str:
    # Remove excessive spaces between single letters like "H I V" → "HIV"
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b", r"\1\2\3", text)
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\b", r"\1\2", text)
    
    # Add missing spaces between joined words like "Whatisthe" → "What is the"
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    
    # Normalize multiple spaces to one
    text = re.sub(r"\s+", " ", text).strip()
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

    # Extract sentences ending with a '?'
    potential_questions = re.findall(r"([A-Z][^?.!]{3,120}\?)", context)
    cleaned, seen = [], set()

    for q in potential_questions:
        # Remove extra prefixes or dataset tags
        q = re.sub(r"(?i)\bTopic\s*\d+[:.\-]?\s*", "", q)
        q = re.sub(r"(?i)\bQ\d+[:.\-]?\s*", "", q)
        q = re.sub(r"PMC\d+/?", "", q)
        q = re.sub(r"HIVChatbot Dataset", "", q, flags=re.I)

        q = fix_spacing(q)

        # Ensure it ends with '?'
        if not q.endswith("?"):
            q += "?"

        # Filter valid unique questions
        if 10 < len(q) < 120 and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q[0].upper() + q[1:])

    # Rank by relevance
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    ranked = sorted(
        cleaned,
        key=lambda x: max(similarity(x, query), similarity(x, answer)),
        reverse=True
    )

    # Remove duplicates across pages (e.g., “What is HIV?” twice)
    unique_ranked = []
    for q in ranked:
        if not any(SequenceMatcher(None, q.lower(), existing.lower()).ratio() > 0.9 for existing in unique_ranked):
            unique_ranked.append(q)

    # Fallback if nothing found
    fallback = [
        "What are the common symptoms of HIV?",
        "How can HIV be prevented?",
        "Can HIV be transmitted through casual contact?",
        "How is HIV treated?",
        "Where can I get tested for HIV?"
    ]

    return unique_ranked[:5] or fallback
