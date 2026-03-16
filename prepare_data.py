from rag_utils import load_job_page

url = "https://jobs.apple.com/en-us/details/200634945-0836/manager-engineering-program-management-aiml-data-operations?team=MLAI"

# ---- CALL SYNC FUNCTION DIRECTLY ----
text = load_job_page(url)

with open("job_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved job_text.txt with length:", len(text))
