
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load values from the .env file
load_dotenv()


def explain_results(question, verified_results):
    """
    Explain manufacturing data using Inkling through OpenRouter.
    The numerical results are calculated by SQL before reaching this function.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing from your .env file."
        )

    # Connect to OpenRouter, not directly to OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Prepare the question and database results
    user_message = (
        f"Manager's question: {question}\n\n"
        "Verified manufacturing database results:\n"
        f"{json.dumps(verified_results, default=str)}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
                You are a manufacturing data analyst.

                Explain the supplied verified data in simple language.
                Use only the supplied results for numerical claims.
                Distinguish observed facts from possible causes.
                Never invent numbers or claim causation without evidence.
                If the data is insufficient, say so.
                Suggest useful next analyses when appropriate.
                """
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    if not answer:
        return "The AI returned an empty response."

    return answer
