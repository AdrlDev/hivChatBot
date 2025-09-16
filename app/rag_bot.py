from fastapi import FastAPI
import os
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

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

from langchain.prompts import PromptTemplate

def get_chatbot():
    vectorstore = get_vectorstore()
    chat = ChatCohere(model="command-a-03-2025", temperature=0, cohere_api_key=api_key)  # type: ignore

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    # Define a proper prompt template
    template = """You are an HIV information assistant. 
Use the provided context to answer the user’s question briefly and clearly. 
    If the answer is not in the context, just say you don’t know.

    Context: {context}
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},  # 👈 pass prompt as dict
        return_source_documents=True
    )
    return qa

# ✅ Function to generate 5 suggested questions
def generate_suggested_questions(query: str, answer: str) -> list[str]:
    # Load the vectorstore retriever
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    # Get context from PDFs related to the query + answer
    docs = retriever.get_relevant_documents(query + " " + answer)
    context = "\n".join([doc.page_content for doc in docs])

    chat = ChatCohere(
        model="command-a-03-2025",
        temperature=0.3,
        cohere_api_key=api_key # type: ignore
    ) # type: ignore
    
    prompt = f"""
    The user asked: "{query}"
    The chatbot answered: "{answer}"

    Here is additional context from HIV information documents:
    {context}

    Based ONLY on this context, generate 5 related follow-up questions
    that are concise, natural, and helpful.
    Return them as plain text questions only, no numbering, no formatting, no markdown.
    """

    response = chat.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    # Split and clean up questions
    questions = [q.strip(" -*•.").replace("**", "") for q in text.split("\n") if q.strip()] # type: ignore

    # Filter out unwanted intro lines
    questions = [q for q in questions if not q.lower().startswith("here are")]

    return questions[:5]

