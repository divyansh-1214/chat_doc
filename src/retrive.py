from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        collection_name="oldmansea",
        embedding_function=embeddings,
        persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db"),
    )

    results = vector_store.similarity_search(
        "What happens to Santiago while fishing?",
        k=3,
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
