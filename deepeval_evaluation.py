from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import GPTModel
import os


def evaluate_email(query, contexts, email):

    judge_model = GPTModel(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=email,
        retrieval_context=contexts
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model=judge_model
    )

    relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model=judge_model
    )

    results = evaluate(
    test_cases=[test_case],
    metrics=[faithfulness, relevancy]
    )

    scores = {}

    for test in results.test_results:
        for metric in test.metrics_data:
            scores[metric.name] = metric.score

    return scores