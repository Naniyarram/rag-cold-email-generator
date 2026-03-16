import asyncio
from rag_utils import load_job_page

url = "https://jobs.apple.com/en-us/details/200634945-0836/manager-engineering-program-management-aiml-data-operations?team=MLAI"

text = asyncio.run(load_job_page(url))

print("Loaded characters:", len(text))
print(text[:500])
