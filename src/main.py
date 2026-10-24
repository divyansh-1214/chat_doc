from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(f"Page {page_number}:\n{text}")

    return "\n\n".join(pages)


def split_text(text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    return splitter.create_documents([text])


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def get_embeddings(chunks: list[Document]) -> list[list[float]]:
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    return vectors


if __name__ == "__main__":
    pdf_text = load_pdf("/home/divyansh/project/chat_doc/src/oldmansea.pdf")
    chunks = split_text(pdf_text)
    print("Total chunks:", len(chunks))

    vectors = get_embeddings(chunks)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="oldmansea",
        persist_directory="./chroma_db",
    )
    print("Length:", len(vectors))
    print("Content:", vectors[:10])
    print("Persisted to:", "./chroma_db")
