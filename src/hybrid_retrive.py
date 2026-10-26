from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def hybrid_search(vector_store, query, k=3):
    stored = vector_store.get()
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored["documents"], stored["metadatas"])
        if text
    ]

    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            semantic_retriever,
            bm25_retriever,
        ],
        weights=[0.5, 0.5],
        c=60,
    )

    return hybrid_retriever.invoke(query)


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        collection_name="oldmansea",
        embedding_function=embeddings,
        persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db"),
    )

    results = hybrid_search(
        vector_store, "What happens to Santiago while fishing?", k=3
    )

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:200])


if __name__ == "__main__":
    main()
