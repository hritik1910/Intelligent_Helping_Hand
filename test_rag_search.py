
from src.rag_search import load_documents, search_documents

# First, check whether the documents were loaded.
documents = load_documents()

print("Total passages loaded:", len(documents))

for document in documents[:5]:
    print("\nSource:", document["source"])
    print("Location:", document["section"])
    print("Text:", document["text"][:200])

# Now search for information relevant to a question.
question = (
    "What should I check when the SPI paste-height "
    "measurement is outside its expected range?"
)

results = search_documents(question)

print("\n========== SEARCH RESULTS ==========")

if not results:
    print("No searchable document text was found.")

else:
    for result in results:
        print("\nSource:", result["source"])
        print("Location:", result["section"])
        print("Similarity:", round(result["similarity"], 3))
        print("Passage:", result["text"])
