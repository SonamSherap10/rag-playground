from operator import itemgetter
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


def create_kb():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={"source": "langchain_knowledge_base.md"}
    )

    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_data"
    )

    return vector_store


def demo_basic_rag():
    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using the context below.
If the answer is not present in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangGraph?",
        "Who created LangChain?",
        "What is LangChain used for?",
        "What is the height of Mount Everest?"
    ]

    print("Basic RAG Demo")

    for question in questions:
        result = rag_chain.invoke({"question": question})

        print(f"Question: {question}")
        print(f"Answer: {result}")
        print()


if __name__ == "__main__":
    demo_basic_rag()
