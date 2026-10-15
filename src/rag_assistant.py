
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.rag_search import search_documents

load_dotenv()


def answer_from_documents(question):
    """
    Search local SOP documents and use the retrieved
    passages to generate a grounded answer.
    """

    passages = search_documents(question)

    if not passages:
        return (
            "I could not find any technical documents. "
            "Please check the docs/sops folder."
        )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing from .env."
        )

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free"
    )

    # Prepare the sources for the AI model.
    context_parts = []

    for passage in passages:
        context_parts.append(
            f"Source: {passage['source']}\n"
            f"Section: {passage['section']}\n"
            f"Passage: {passage['text']}"
        )

    retrieved_context = "\n\n".join(context_parts)

    # Connect to the model through OpenRouter.
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
                You are a technical document assistant.

                Answer the user's question using the supplied
                document passages.

                Explain the information in simple language.
                Do not invent procedures or unsupported facts.
                Cite the source filename and section.
                If the documents do not contain enough
                information, clearly say so.

                The documents may be fictional samples.
                Never present them as officially approved
                equipment instructions.
                """
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved document passages:\n"
                    f"{retrieved_context}"
                )
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    if not answer:
        return "The AI returned an empty response."

    return answer
