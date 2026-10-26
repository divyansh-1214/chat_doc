import argparse
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from hybrid_retrive import hybrid_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "src" / "oldmansea.pdf"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "oldmansea"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_llm() -> ChatOpenAI:
    # model = os.getenv("CHAT_MODEL")
    # if not model:
    #     raise RuntimeError(
    #         "Set CHAT_MODEL to the model name served by your local OpenAI-compatible API."
    #     )
    return ChatOpenAI(
        base_url="http://localhost:1235/v1",
        api_key="local",
        model="YOUR_MODEL_ID",
        temperature=0.2,
        max_tokens=500,
    )


def load_pdf(file_path: Path) -> list[Document]:
    reader = PdfReader(str(file_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": page_number},
                )
            )

    if not documents:
        raise ValueError(f"No extractable text found in PDF: {file_path}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    return splitter.split_documents(documents)


def create_vector_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    chunks = split_documents(load_pdf(PDF_PATH))
    ids = [f"{PDF_PATH.name}-{index}" for index in range(len(chunks))]

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
        ids=ids,
    )
    print(f"Indexed {len(chunks)} chunks in {CHROMA_PATH}")
    return vector_store


def answer_question(vector_store: Chroma, query: str, llm: ChatOpenAI) -> str:
    documents = hybrid_search(vector_store, query, k=3)
    if not documents:
        raise RuntimeError(
            "No relevant documents found. Index the PDF first with --ingest."
        )

    context = "\n\n".join(
        f"[{document.metadata.get('source', 'unknown')}, "
        f"page {document.metadata.get('page', 'unknown')}]\n"
        f"{document.page_content}"
        for document in documents
    )
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    # return str(llm.invoke(prompt))
    return str(llm.invoke(prompt).content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the indexed PDF.")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Extract, split, and index the source PDF into Chroma.",
    )
    parser.add_argument(
        "--query", help="Question to ask; prompts interactively if omitted."
    )
    args = parser.parse_args()

    embeddings = create_embeddings()
    if args.ingest:
        create_vector_db(embeddings)
        return

    query = (args.query or input("Enter your query: ")).strip()
    if not query:
        raise ValueError("The query cannot be empty.")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    if not vector_store.get(limit=1)["ids"]:
        raise RuntimeError(
            "The PDF is not indexed. Run this script with --ingest first."
        )

    print(answer_question(vector_store, query, create_llm()))


if __name__ == "__main__":
    main()
