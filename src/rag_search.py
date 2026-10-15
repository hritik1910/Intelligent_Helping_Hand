
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Folder containing all your technical documents
DOCUMENT_FOLDER = Path("docs/sops")

# This model converts text into numerical representations.
# The first run may download the model.
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def split_into_chunks(text, chunk_size=180, overlap=30):
    """
    Divide a document into smaller passages.

    chunk_size: maximum number of words per passage
    overlap: number of words shared between passages
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def load_documents():
    """
    Read all TXT and PDF files from docs/sops.
    Return text passages and their source locations.
    """

    records = []

    if not DOCUMENT_FOLDER.exists():
        raise FileNotFoundError(
            f"Document folder not found: {DOCUMENT_FOLDER}"
        )

    # Read each supported file in the folder.
    for path in sorted(DOCUMENT_FOLDER.iterdir()):

        # Ignore unsupported files and subfolders.
        if not path.is_file():
            continue

        extension = path.suffix.lower()

        # --------------------------------------
        # CASE 1: Read TXT files
        # --------------------------------------
        if extension == ".txt":

            text = path.read_text(encoding="utf-8")

            chunks = split_into_chunks(text)

            for number, chunk in enumerate(chunks, start=1):
                records.append({
                    "source": path.name,
                    "section": f"section {number}",
                    "text": chunk
                })

        # --------------------------------------
        # CASE 2: Read PDF files
        # --------------------------------------
        elif extension == ".pdf":

            reader = PdfReader(str(path))

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):
                # Extract selectable text from this PDF page.
                page_text = page.extract_text() or ""

                if not page_text.strip():
                    continue

                chunks = split_into_chunks(page_text)

                for number, chunk in enumerate(chunks, start=1):
                    records.append({
                        "source": path.name,
                        "section": (
                            f"page {page_number}, "
                            f"chunk {number}"
                        ),
                        "text": chunk
                    })

    return records


def search_documents(question, top_k=3):
    """
    Return the document passages most relevant to a question.
    """

    records = load_documents()

    if not records:
        return []

    # Convert every passage into an embedding.
    document_vectors = embedding_model.encode(
        [record["text"] for record in records],
        normalize_embeddings=True
    )

    # Convert the user's question into an embedding.
    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Compare the question with the document passages.
    similarity_scores = document_vectors @ question_vector

    # Select the highest-scoring passages.
    number_to_return = min(top_k, len(records))

    best_indices = np.argsort(similarity_scores)[::-1][
        :number_to_return
    ]

    results = []

    for index in best_indices:
        result = dict(records[index])
        result["similarity"] = float(similarity_scores[index])
        results.append(result)

    return results
