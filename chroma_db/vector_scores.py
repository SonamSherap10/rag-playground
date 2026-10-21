from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tempfile

# Use a local Ollama embedding model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        print(
            f"Vector store created with "
            f"{vectorstore._collection.count()} documents."
        )

        query = "What is LangChain?"

        results = vectorstore.similarity_search(
            query,
            k=2,
        )

        print(f"\nTop 2 results for: {query}")

        for i, doc in enumerate(results, start=1):
            print(
                f"Result {i}: {doc.page_content} "
                f"(Source: {doc.metadata['source']})"
            )


def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "Explain vector stores."

        results = vectorstore.similarity_search_with_score(
            query,
            k=3,
        )

        print(f"\nTop 3 results for: {query}")

        for i, (doc, distance) in enumerate(results, start=1):
            print(
                f"Result {i}: {doc.page_content}\n"
                f"Distance: {distance:.4f}\n"
                f"Source: {doc.metadata['source']}\n"
            )

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "What databases are available?"

       #without filtering
        results = vectorstore.similarity_search(
            query,
            k=5,
        )
        print(f"\nTop 5 results for: {query} without filtering")
        for i, doc in enumerate(results, start=1):
            print(
                f"Result {i}: {doc.page_content} "
                f"(Source: {doc.metadata['source']})"
            )


        # Filter by topic
        filter_criteria = {"topic": "database"}

        results = vectorstore.similarity_search(
            query,
            k=5,
            filter=filter_criteria,
        )

        print(f"\nTop 5 results for: {query} with filter {filter_criteria}")

        for i, doc in enumerate(results, start=1):
            print(
                f"Result {i}: {doc.page_content} "
                f"(Source: {doc.metadata['source']})"
            )


if __name__ == "__main__":
    chroma_basics()
    similarity_search_with_scores()
    metadata_filtering()
