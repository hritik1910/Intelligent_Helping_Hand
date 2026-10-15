
from src.rag_assistant import answer_from_documents

question = (
    "What should I check when the SPI paste-height "
    "measurement is outside the expected range?"
)

answer = answer_from_documents(question)

print("\nQuestion:")
print(question)

print("\nAI answer:")
print(answer) 
