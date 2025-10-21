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
from langchain import LLMChain, PromptTemplate
from langchain.chains.llm import LLMChain
import json
import logging

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

# small, robust text cleaner for fallback
def strong_clean_text(s: str) -> str:
    # fix spaced single letters: "H I V" -> "HIV"
    s = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3', s)
    s = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', s)

    # insert space before capital when previous is lowercase: "attackThe" -> "attack The"
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)

    # break common glued lowercase sequences heuristically:
    # "attackthebody" -> "attack the body" by inserting spaces before common small words
    small_words = [
        'the','a','an','and','or','but','to','of','in','for','on','with','as','by','from',
        'is','are','was','were','be','been','do','does','did','this','that','these','those',
        'have','has','had','when','where','how','what','why','which','who'
    ]
    # naive insertion: try to insert spaces in runs of lowercase letters if they contain a small_word boundary
    def insert_small_word_spaces(text):
        for w in small_words:
            # look for patterns like 'attackthe' or 'whendo' and make them 'attack the'
            pattern = r'([a-z]{3,})(' + re.escape(w) + r')([a-z]{0,})'
            text = re.sub(pattern, r'\1 \2 \3', text)
        return text

    s = insert_small_word_spaces(s)

    # ensure keywords like HIV/AIDS are uppercase and joined: "H I V" -> "HIV", "hiv" -> "HIV"
    s = re.sub(r'\b[hH]\s*[iI]\s*[vV]\b', 'HIV', s)
    s = re.sub(r'\b[aA]\s*[iI]\s*[dD]\s*[sS]\b', 'AIDS', s)

    # normalize spacing and punctuation
    s = re.sub(r'\s+([,?.!;:])', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # fix weird "attackthe" where prior heuristics missed: split lowercase runs by looking for English-like boundaries
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)

    # capitalize first character
    if s:
        s = s[0].upper() + s[1:]

    return s

# Primary: generate questions using LLM (ChatCohere via your existing ChatCohere wrapper)
def generate_suggested_questions(query: str, answer: str) -> list[str]:
    """
    Primary approach: Ask the LLM to produce 5 concise, user-friendly follow-up questions
    based on the answer and top retrieved document text (so we avoid relying on broken PDF question extraction).
    Fallback: aggressive cleaning and extraction from PDFs if LLM fails.
    """

    vectorstore = get_vectorstore()
    # get top N docs for context
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 6, "score_threshold": 0.25}
    )
    docs = retriever.get_relevant_documents(answer)
    # keep only our two PDF sources to be consistent
    docs = [
        d for d in docs
        if any(x in d.metadata.get("source", "").lower()
               for x in ["hiv_qa.pdf", "hiv_information_sheets.pdf"])
    ]

    context_text = "\n\n".join([d.page_content for d in docs])[:4000]  # cap length to avoid huge prompts

    # LLM prompt: ask for 5 short follow-up questions, no analysis, output as JSON array
    prompt_text = """You are an assistant that generates concise, friendly follow-up questions for a user about HIV.
                    Given the short user query and the authoritative answer and the supporting context (documents), produce exactly 5 distinct, clear, short follow-up questions that a chatbot could present as suggested next questions.
                    Requirements:
                    - Questions must be about HIV-related topics or helpful next steps (testing, symptoms, prevention, treatment, sources, timeline).
                    - Avoid quoting the context verbatim; compose natural questions.
                    - Return the output as a JSON array of strings only (no extra text).
                    - Keep each question short (<= 80 characters) and properly spaced, capitalized, and punctuated.
                    User Query:
                    \"\"\"{query}\"\"\"
                    Answer:
                    \"\"\"{answer}\"\"\"
                    Context (top documents, for reference):
                    \"\"\"{context}\"\"\"
                    Now produce the JSON array of 5 questions.
                    """

    prompt = PromptTemplate(
        input_variables=["query", "answer", "context"],
        template=prompt_text
    )

    # Try to call your ChatCohere LLM (langchain wrapper)
    try:
        chat = ChatCohere(model="command-a-03-2025", temperature=0.0, cohere_api_key=api_key) # pyright: ignore[reportArgumentType]
        chain = LLMChain(llm=chat, prompt=prompt)
        resp = chain.run({"query": query, "answer": answer, "context": context_text})
        # resp should be a JSON array, but LLM sometimes returns with newlines; try to extract JSON
        resp = resp.strip()
        # Allow for cases where model returns triple-backticks or explanation; extract first JSON array-looking substring
        m = re.search(r'(\[[\s\S]*?\])', resp)
        if m:
            arr_text = m.group(1)
        else:
            # assume entire response is the array
            arr_text = resp

        questions = json.loads(arr_text)
        # sanitize and ensure exactly 5 unique clean items
        cleaned = []
        seen = set()
        for q in questions:
            if not isinstance(q, str):
                continue
            q_clean = q.strip()
            q_clean = re.sub(r'\s+', ' ', q_clean)
            q_clean = q_clean[0].upper() + q_clean[1:] if q_clean else q_clean
            if not q_clean.endswith('?'):
                q_clean += '?'
            if q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                cleaned.append(q_clean)
            if len(cleaned) >= 5:
                break

        if len(cleaned) >= 1:
            # pad with sensible fallbacks if <5
            fallback = [
                "What are the common symptoms of HIV?",
                "How can HIV be prevented?",
                "Where can I get tested for HIV?",
                "How is HIV treated?",
                "Is HIV the same as AIDS?"
            ]
            for f in fallback:
                if len(cleaned) >= 5:
                    break
                if f.lower() not in seen:
                    cleaned.append(f)
            return cleaned[:5]

    except Exception as e:
        # log and fall back to extraction-based approach
        logging.exception("LLM-based suggested questions failed, falling back to extraction. Error: %s", e)

    # --------------------------
    # Fallback: extract and aggressively clean questions from PDF text
    # --------------------------
    try:
        context = "\n".join([d.page_content for d in docs])
        # attempt to normalize joins before regex
        context = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', context)   # split camel-case joins
        context = re.sub(r'\s+', ' ', context)
        # extract candidate question-like substrings (longer capture window)
        potential_questions = re.findall(r'([A-Z][A-Za-z0-9 ,;:\'\"()\/\-]{3,240}\?)', context)
        cleaned = []
        seen = set()
        for q in potential_questions:
            q = re.sub(r'(?i)\bTopic\s*\d+[:.\-]?\s*', '', q)
            q = re.sub(r'(?i)\bQ\d+[:.\-]?\s*', '', q)
            q = re.sub(r'PMC\d+/?', '', q)
            q = re.sub(r'HIVChatbot Dataset', '', q, flags=re.I)
            q = strong_clean_text(q)
            # additional small grammar fixes
            q = re.sub(r'\bWhatare signsof\b', 'What are the signs of', q, flags=re.I)
            q = re.sub(r'\bWhendo\b', 'When do', q, flags=re.I)
            q = re.sub(r'\bWhatis\b', 'What is', q, flags=re.I)
            q = re.sub(r'\bIs HIVthesameas\b', 'Is HIV the same as', q, flags=re.I)
            q = re.sub(r'\s+', ' ', q).strip()
            if not q.endswith('?'):
                q += '?'
            q = q[0].upper() + q[1:]
            if len(q) > 8 and len(q) < 120 and q.lower() not in seen:
                seen.add(q.lower())
                cleaned.append(q)
            if len(cleaned) >= 10:
                break

        # dedupe near-duplicates
        unique_ranked = []
        for q in cleaned:
            if not any(SequenceMatcher(None, q.lower(), r.lower()).ratio() > 0.9 for r in unique_ranked):
                unique_ranked.append(q)

        # final selection: top 5 or fallback
        result = unique_ranked[:5]
        if len(result) < 5:
            fallback = [
                "What are the common symptoms of HIV?",
                "How can HIV be prevented?",
                "Where can I get tested for HIV?",
                "How is HIV treated?",
                "Is HIV the same as AIDS?"
            ]
            for f in fallback:
                if f.lower() not in [x.lower() for x in result]:
                    result.append(f)
                if len(result) >= 5:
                    break
        return result[:5]
    except Exception as e:
        logging.exception("Final fallback also failed: %s", e)
        # ultimate fallback static questions
        return [
            "What are the common symptoms of HIV?",
            "How can HIV be prevented?",
            "Can HIV be transmitted through casual contact?",
            "How is HIV treated?",
            "Where can I get tested for HIV?"
        ]
