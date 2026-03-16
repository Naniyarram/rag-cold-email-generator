# rag-cold-email-generator
Retrieval-Augmented Generation (RAG) application that generates personalized cold emails from job descriptions. Uses SentenceTransformers embeddings, vector search with ChromaDB, and LLaMA-based LLMs to produce context-aware outreach emails.



# RAG Cold Email Generator

AI-powered Cold Email Generator built using Retrieval-Augmented Generation (RAG).  
The system analyzes job descriptions and generates personalized outreach emails by retrieving relevant context and generating responses using LLMs.

This project demonstrates an end-to-end **RAG pipeline with semantic retrieval, vector databases, and LLM generation**, showcasing practical applications of Generative AI.

---

# Project Overview

Cold outreach emails are commonly used for networking, job opportunities, and collaborations. Writing personalized emails for each opportunity is time-consuming.

This project automates that process using **Retrieval-Augmented Generation (RAG)**.

The system:
1. Extracts information from job descriptions
2. Retrieves relevant context using embeddings
3. Generates personalized cold emails using an LLM

RAG enhances LLM outputs by retrieving external information and using it as context before generating responses. This helps produce more accurate and contextually relevant outputs. 

---

# Features

- Retrieval-Augmented Generation pipeline
- Semantic search using embeddings
- Vector database for efficient retrieval
- LLM-powered email generation
- Interactive Streamlit interface
- Evaluation metrics for response quality

---

# System Architecture

The pipeline follows a standard RAG architecture:


Job Description
│
▼
Text Processing / Chunking
│
▼
Embedding Generation (SentenceTransformers)
│
▼
Vector Database (Chroma)
│
▼
Semantic Retrieval
│
▼
LLM Generation (LLaMA / Groq)
│
▼
Personalized Cold Email


---

# Tech Stack

**Programming Language**
- Python

**Machine Learning / NLP**
- SentenceTransformers
- Embeddings
- Semantic Retrieval

**LLM**
- LLaMA / Groq API

**Vector Database**
- ChromaDB

**Frameworks / Tools**
- Streamlit
- LangChain
- BeautifulSoup
- Playwright

---

# RAG Pipeline Explanation

The system implements a standard RAG workflow:

### 1. Data Ingestion
Job descriptions are collected and processed into structured text.

### 2. Text Chunking
The text is split into smaller chunks to improve retrieval performance.

### 3. Embedding Generation
SentenceTransformers converts text chunks into dense vector embeddings.

### 4. Vector Storage
Embeddings are stored in a **Chroma vector database** for similarity search.

### 5. Semantic Retrieval
Relevant text chunks are retrieved based on similarity with the user query.

### 6. Response Generation
The retrieved context is passed to the LLM to generate a personalized cold email.

---

# Evaluation Metrics

To evaluate the quality of the RAG system, two important metrics are used:

### Faithfulness

Faithfulness measures whether the generated response is grounded in the retrieved context without introducing unsupported information or hallucinations.

Higher faithfulness means the model's output accurately reflects the retrieved knowledge.

Score range:

0 → hallucinated response
1 → fully grounded response


### Answer Relevance

Answer relevance measures how well the generated answer addresses the user's query.

It penalizes responses that:
- contain irrelevant information
- fail to answer the question properly

Score range:

0 → irrelevant answer
1 → highly relevant answer


These metrics help evaluate both **accuracy and usefulness** of the generated responses.

---

# Example Workflow

Example input:

url: https://endee.io/careers/machine-learning-intern-mkwnd6wk

Generated Output:
<img width="1920" height="1014" alt="Screenshot 2026-03-16 223102" src="https://github.com/user-attachments/assets/5ce1af88-6f16-43b8-b95b-9fe25413357d" />
<img width="1920" height="1011" alt="Screenshot 2026-03-16 223127" src="https://github.com/user-attachments/assets/5468f9a6-1fc7-4b06-bef1-62614c8f36bf" />


