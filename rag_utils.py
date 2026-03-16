# ---------- STEP 1: LOAD & CLEAN WEBPAGE ----------
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def load_job_page(url: str) -> str:
    """Robust loader with JS wait + graceful failure."""

    # ---------- TRY SIMPLE REQUEST ----------
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)

        if r.status_code == 200 and len(r.text) > 2000:
            html = r.text
        else:
            html = None
    except Exception:
        html = None

    # ---------- PLAYWRIGHT FALLBACK ----------
    if not html:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # wait for network + JS render
                page.goto(url, timeout=60000, wait_until="networkidle")

                # give extra render time for Workday-like sites
                page.wait_for_timeout(5000)

                html = page.content()
                browser.close()
        except Exception:
            return ""  # graceful failure

    # ---------- CLEAN ----------
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text



# ================= CACHE-AWARE LOADER =================
import os
import hashlib


def load_or_scrape_job(url: str) -> str:
    """
    Load job text from cache if available,
    otherwise scrape once and store.
    """

    os.makedirs("cache", exist_ok=True)

    # create unique filename from URL
    filename = hashlib.md5(url.encode()).hexdigest() + ".txt"
    path = os.path.join("cache", filename)

    # ---------- LOAD FROM CACHE ----------
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ---------- SCRAPE ----------
    text = load_job_page(url)

    # ---------- SAVE TO CACHE ----------
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return text



# ---------- STEP 2: TEXT CHUNKING ----------
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# ---------- STEP 3: CREATE VECTOR STORE ----------

from chromadb import Client
from chromadb.utils import embedding_functions


def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No text chunks found. Cannot create vector store.")

    client = Client()

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="jobs",
        embedding_function=embedding_function
    )

    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )

    return collection




    # ---------- STEP 4: RETRIEVAL ----------
def retrieve_relevant_chunks(collection, query: str, k: int = 3):
    """Retrieve top-k relevant chunks from Chroma."""
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0]


# ---------- STEP 5: GROQ EMAIL GENERATION ----------
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_groq import ChatGroq

def generate_cold_email(context_chunks, user_query):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant that writes professional cold emails.

Using the job description below, write a concise cold email expressing
interest in the role and highlighting relevant AI/ML and data skills.

Job Description:
{context}

User intent:
{user_query}

Write only the final email.
"""

    response = llm.invoke(prompt)
    return response.content


