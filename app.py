#  WINDOWS EVENT LOOP FIX 
import sys
import asyncio
from deepeval_evaluation import evaluate_email

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


import streamlit as st

from rag_utils import (
    load_or_scrape_job,
    chunk_text,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_cold_email,
)

#  PAGE CONFIG 
st.set_page_config(
    page_title="AI Cold Email Generator (RAG)",
    page_icon="✉️",
    layout="centered",
)

st.title("✉️ AI Cold Email Generator")
st.caption(
    "Generate personalized cold emails from any job posting using "
    "Retrieval-Augmented Generation (RAG) + Groq LLM."
)

#  INPUTS
job_url = st.text_input("🔗 Enter Job URL")

user_intent = st.text_input(
    "📝 Your intent",
    value="Write a professional cold email applying for this AI/ML role",
)

manual_text = st.text_area(
    "📄 If scraping fails, paste the job description here (optional)",
    height=200,
)

generate_btn = st.button("🚀 Generate Cold Email")

# MAIN FLOW 
if generate_btn:

    #  Step 1: Get text 
    text = ""

    if job_url:
        with st.spinner("🔎 Attempting to load job page..."):
            text = load_or_scrape_job(job_url)

    #  Step 2: Fallback to manual paste
    if not text or len(text.strip()) == 0:
        if manual_text and len(manual_text.strip()) > 0:
            st.warning("⚠️ Scraping failed. Using manually pasted job description.")
            text = manual_text
        else:
            st.error(
                "❌ Could not extract text from the URL.\n\n"
                "👉 Please paste the job description manually in the box above."
            )
            st.stop()

    st.write("**Text length:**", len(text))

    #  Step 3: Chunking 
    with st.spinner("✂️ Splitting text into chunks..."):
        chunks = chunk_text(text)

    if not chunks:
        st.error("Chunking failed. No usable content found.")
        st.stop()

    # Step 4: Vector DB 
    with st.spinner("🧠 Creating embeddings & vector database..."):
        collection = create_vector_store(chunks)

    #  Step 5: Retrieval 
    with st.spinner("📚 Retrieving relevant context..."):
        relevant_chunks = retrieve_relevant_chunks(collection, user_intent)

    if not relevant_chunks:
        st.error("Could not retrieve relevant context.")
        st.stop()

    # Step 6: Generation 
    with st.spinner("✍️ Generating cold email using Groq..."):
        email = generate_cold_email(relevant_chunks, user_intent)

    contexts = [chunk for chunk in relevant_chunks]

    scores = evaluate_email(
        query=user_intent,
        contexts=contexts,
        email=email
    )

    #  OUTPUT 
    st.success("✅ Cold email generated!")

    st.subheader("📧 Generated Cold Email")

    st.text_area("", email, height=300)

    st.caption("You can copy this email and send it via Gmail or LinkedIn.")

    st.subheader("📊 Evaluation Metrics")

    st.metric(
        "Faithfulness",
        f"{scores.get('Faithfulness', 0):.2f}"
    )

    st.metric(
        "Answer Relevancy",
        f"{scores.get('Answer Relevancy', 0):.2f}"
    )
