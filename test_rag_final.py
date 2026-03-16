import asyncio
from rag_utils import (
    load_job_page,
    chunk_text,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_cold_email,
)

url = "https://jobs.apple.com/en-us/details/200634945-0836/manager-engineering-program-management-aiml-data-operations?team=MLAI"
query = "Write a cold email applying for this AI/ML role"

#  Load + clean text
text = asyncio.run(load_job_page(url))

# Chunk
chunks = chunk_text(text)

#  Vector DB
collection = create_vector_store(chunks)

# Retrieve relevant chunks
relevant_chunks = retrieve_relevant_chunks(collection, query)

# Generate cold email
email = generate_cold_email(relevant_chunks, query)

print("\n===== GENERATED COLD EMAIL =====\n")
print(email)
