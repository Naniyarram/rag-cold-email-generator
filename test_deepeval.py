from deepeval_evaluation import evaluate_rag_pipeline, print_results


query = "Write a professional cold email applying for this AI Engineer role"

contexts = [
    "The company is hiring an AI Engineer experienced in Python, ML, and LLMs.",
    "The role involves building AI products and working with modern ML frameworks."
]

generated_email = """
Hello Hiring Manager,

I recently came across your AI Engineer position and was excited to see the
focus on machine learning and LLM technologies. With my experience in Python,
ML models, and building AI-powered systems, I believe I could contribute
effectively to your team.

I would welcome the opportunity to discuss how my background aligns with
your needs.

Best regards,
Nani
"""

results = evaluate_rag_pipeline(query, contexts, generated_email)

print_results(results)