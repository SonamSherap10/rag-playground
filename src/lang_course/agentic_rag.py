
"""
Agentic RAG using LangGraph + Chroma + Ollama.

Workflow:
    START
      |
    retrieve
      |
    grade_documents
      |
      +---- relevant --------> generate_answer ----> END
      |
      +---- not relevant ----> rewrite_query
                                  |
                               retrieve
                                  |
                              grade_documents
                                  |
                         retry limit reached?
                                  |
                               fallback
                                  |
                                 END

Local models:
    Embeddings: nomic-embed-text
    Chat model: llama3.2:3b

"""

from __future__ import annotations

import hashlib
import logging

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, START, StateGraph


# ============================================================
# 1. CONFIGURATION
# ============================================================

COLLECTION_NAME = "technical_knowledge"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3.2:3b"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 80

TOP_K = 4

MAX_RETRIEVAL_ATTEMPTS = 2


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. SAMPLE KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = [
    Document(
        page_content="""
        Orion Billing Platform

        Orion is a subscription billing platform.
        It processes customer payments, generates invoices,
        and maintains subscription records.

        Deployment and Rollback

        Orion uses canary deployments.
        A new release initially receives five percent
        of production traffic.

        The deployment controller monitors HTTP 5xx errors,
        payment success rates, and application latency.

        If the HTTP 5xx error rate exceeds two percent
        for five consecutive minutes, traffic is automatically
        rolled back to the previous stable release.

        A rollback is also triggered if payment success
        decreases by three percentage points compared
        with the stable release.

        Invoice Processing

        The invoice worker retries failed payments
        after one minute, five minutes, and fifteen minutes.

        After three failed attempts, an invoice is
        marked for manual review.
        """,
        metadata={
            "source": "orion_billing.md",
            "category": "billing",
        },
    ),

    Document(
        page_content="""
        Atlas Identity Service

        Atlas manages authentication, authorization,
        user sessions, and access tokens.

        Access tokens expire after fifteen minutes.

        Refresh tokens are rotated whenever a new
        access token is issued.

        Security

        Passwords are hashed using Argon2id.

        Rate limiting is enabled on login and
        password reset endpoints.

        If the authentication database is unavailable,
        Atlas returns HTTP 503 and does not issue
        new access tokens.
        """,
        metadata={
            "source": "atlas_identity.md",
            "category": "identity",
        },
    ),

    Document(
        page_content="""
        Nova Analytics Platform

        Nova processes event streams and generates
        analytical reports.

        Apache Kafka is used for event ingestion.

        Failed events are retried three times.

        After repeated failures, events are moved
        into a dead-letter queue for inspection.

        Aggregated results are stored in PostgreSQL.
        Raw events are retained in object storage
        for thirty days.
        """,
        metadata={
            "source": "nova_analytics.md",
            "category": "analytics",
        },
    ),
]


# ============================================================
# 4. CREATE MODELS
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    num_predict=512,
    timeout=120,
)


# ============================================================
# 5. VECTOR STORE INITIALIZATION
# ============================================================

def create_vector_store() -> Chroma:
    """
    Create or open a persistent Chroma collection.

    Existing documents are retained.
    New documents are added only if their IDs are missing.
    """

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(KNOWLEDGE_BASE)

    documents = []
    ids = []

    for index, chunk in enumerate(chunks):

        source = chunk.metadata["source"]

        raw_id = (
            f"{source}:{index}:{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = index

        documents.append(chunk)
        ids.append(chunk_id)

    # Check which IDs already exist.
    existing = vector_store.get(ids=ids)

    existing_ids = set(existing["ids"])

    new_documents = []
    new_ids = []

    for document, chunk_id in zip(documents, ids):

        if chunk_id not in existing_ids:

            new_documents.append(document)
            new_ids.append(chunk_id)

    if new_documents:

        vector_store.add_documents(
            documents=new_documents,
            ids=new_ids,
        )

        logger.info(
            "Added %d new chunks to Chroma.",
            len(new_documents),
        )

    else:

        logger.info(
            "Knowledge base already indexed. No new chunks."
        )

    logger.info(
        "Vector store contains %d records.",
        vector_store._collection.count(),
    )

    return vector_store


# ============================================================
# 6. STRUCTURED OUTPUT FOR DOCUMENT GRADING
# ============================================================

class RelevanceGrade(BaseModel):
    """
    Structured result returned by the document grader.
    """

    relevant: bool = Field(
        description=(
            "True if the retrieved context contains information "
            "that helps answer the user's question."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the retrieved context "
            "is or is not relevant."
        )
    )


grader = llm.with_structured_output(
    RelevanceGrade,
    method="json_schema",
)


# ============================================================
# 7. GRAPH STATE
# ============================================================

class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Each node reads this state and returns updates.
    """

    question: str

    search_query: str

    documents: list[Document]

    retrieval_scores: list[float]

    relevant: bool

    grade_reason: str

    attempts: int

    answer: str


# ============================================================
# 8. NODE 1: RETRIEVE DOCUMENTS
# ============================================================

def retrieve(state: AgentState) -> dict:
    """
    Search Chroma using the current search query.
    """

    query = state["search_query"]

    attempt = state["attempts"] + 1

    logger.info(
        "Retrieval attempt %d. Query: %s",
        attempt,
        query,
    )

    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=TOP_K,
    )

    documents = []
    scores = []

    for document, score in results:

        documents.append(document)
        scores.append(float(score))

    for index, (document, score) in enumerate(
        zip(documents, scores),
        start=1,
    ):

        logger.info(
            "Rank %d | relevance %.4f | source %s",
            index,
            score,
            document.metadata.get("source"),
        )

    return {
        "documents": documents,
        "retrieval_scores": scores,
        "attempts": attempt,
    }


# ============================================================
# 9. NODE 2: GRADE DOCUMENTS
# ============================================================

def grade_documents(state: AgentState) -> dict:
    """
    Ask the LLM whether retrieved documents
    contain useful evidence for the question.
    """

    question = state["question"]

    documents = state["documents"]

    if not documents:

        return {
            "relevant": False,
            "grade_reason": "No documents were retrieved.",
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are evaluating retrieved documents for a RAG system.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Determine whether the retrieved context contains
information that can help answer the question.

Rules:
- Judge relevance based on evidence in the context.
- Do not use outside knowledge.
- A partially useful document may still be relevant.
- If the context is unrelated, mark it false.
- Do not confuse keyword overlap with actual relevance.

Return the structured relevance judgment.
"""

    result = grader.invoke(prompt)

    logger.info(
        "Document relevance: %s | Reason: %s",
        result.relevant,
        result.reason,
    )

    return {
        "relevant": result.relevant,
        "grade_reason": result.reason,
    }


# ============================================================
# 10. NODE 3: REWRITE QUERY
# ============================================================

def rewrite_query(state: AgentState) -> dict:
    """
    Rewrite the original question to improve retrieval.

    The rewritten query does not replace the original question.
    It is only used for the next vector search.
    """

    question = state["question"]

    previous_query = state["search_query"]

    prompt = f"""
You are improving a search query for a technical
documentation retrieval system.

Original user question:
{question}

Previous search query:
{previous_query}

The previous retrieval did not return sufficiently
relevant documents.

Rewrite the search query to improve retrieval.

Rules:
- Preserve the original intent.
- Use important technical terms.
- Expand abbreviations where useful.
- Do not answer the question.
- Return only the rewritten search query.
"""

    response = llm.invoke(prompt)

    rewritten_query = str(response.content).strip()

    if not rewritten_query:
        rewritten_query = question

    logger.info(
        "Rewritten search query: %s",
        rewritten_query,
    )

    return {
        "search_query": rewritten_query,
    }


# ============================================================
# 11. NODE 4: GENERATE ANSWER
# ============================================================

def generate_answer(state: AgentState) -> dict:
    """
    Generate an answer using retrieved evidence only.
    """

    question = state["question"]

    documents = state["documents"]

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown",
        )

        context_parts.append(
            f"SOURCE: {source}\n"
            f"CONTENT:\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question using only the
provided retrieved context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:
1. Do not invent facts.
2. Do not use knowledge outside the context.
3. If the context only partially answers the question,
   clearly explain what is missing.
4. Mention the source filename when making factual claims.
5. Be clear and concise.

ANSWER:
"""

    response = llm.invoke(prompt)

    answer = str(response.content).strip()

    logger.info("Answer generated.")

    return {
        "answer": answer,
    }


# ============================================================
# 12. NODE 5: FALLBACK
# ============================================================

def fallback(state: AgentState) -> dict:
    """
    Return a safe response when retrieval fails.
    """

    logger.warning(
        "Could not find relevant documents after %d attempts.",
        state["attempts"],
    )

    return {
        "answer": (
            "I couldn't find sufficient relevant information "
            "in the available knowledge base to answer "
            "that question reliably."
        ),
    }


# ============================================================
# 13. CONDITIONAL ROUTING
# ============================================================

def route_after_grading(
    state: AgentState,
) -> Literal[
    "generate_answer",
    "rewrite_query",
    "fallback",
]:
    """
    Decide the next graph node.

    Relevant documents -> generate answer.

    Irrelevant documents with attempts remaining
    -> rewrite query.

    Irrelevant documents with no attempts remaining
    -> fallback.
    """

    if state["relevant"]:

        return "generate_answer"

    if state["attempts"] < MAX_RETRIEVAL_ATTEMPTS:

        return "rewrite_query"

    return "fallback"


# ============================================================
# 14. BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_agent():
    """
    Construct and compile the Agentic RAG graph.
    """

    workflow = StateGraph(AgentState)

    # Register nodes.
    workflow.add_node("retrieve", retrieve)

    workflow.add_node(
        "grade_documents",
        grade_documents,
    )

    workflow.add_node(
        "rewrite_query",
        rewrite_query,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer,
    )

    workflow.add_node("fallback", fallback)

    # Define starting point.
    workflow.add_edge(START, "retrieve")

    # Retrieve -> Grade.
    workflow.add_edge(
        "retrieve",
        "grade_documents",
    )

    # Grade -> Generate / Rewrite / Fallback.
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # Rewrite -> Retrieve.
    workflow.add_edge(
        "rewrite_query",
        "retrieve",
    )

    # Terminal nodes.
    workflow.add_edge(
        "generate_answer",
        END,
    )

    workflow.add_edge(
        "fallback",
        END,
    )

    return workflow.compile()


# ============================================================
# 15. EXECUTE AGENT
# ============================================================

def run_agent(agent, question: str) -> None:

    initial_state: AgentState = {
        "question": question,
        "search_query": question,
        "documents": [],
        "retrieval_scores": [],
        "relevant": False,
        "grade_reason": "",
        "attempts": 0,
        "answer": "",
    }

    result = agent.invoke(initial_state)

    print("\n" + "=" * 75)
    print("AGENTIC RAG RESULT")
    print("=" * 75)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieval attempts:")
    print(result["attempts"])

    print("\nFinal search query:")
    print(result["search_query"])

    print("\nRetrieved sources:")

    for document, score in zip(
        result["documents"],
        result["retrieval_scores"],
    ):

        print(
            f"- {document.metadata['source']} "
            f"(relevance: {score:.4f})"
        )

    print("\nFinal grading reason:")
    print(result["grade_reason"])

    print("=" * 75)


# ============================================================
# 16. MAIN
# ============================================================

def main():

    global vector_store

    logger.info("Initializing Agentic RAG system...")

    vector_store = create_vector_store()

    agent = build_agent()

    logger.info("Agent graph compiled successfully.")

    # Print graph structure as Mermaid.
    print("\nAGENT GRAPH:")
    print(agent.get_graph().draw_mermaid())

    test_queries = [
        "What triggers an automatic rollback in Orion?",
        "How does Atlas handle refresh tokens?",
        "What is the capital of France?",
    ]

    print("\nAvailable test queries:")

    for index, query in enumerate(
        test_queries,
        start=1,
    ):

        print(f"{index}. {query}")

    print("\nEnter a number to run a test query.")
    print("Or type your own question.")
    print("Type 'exit' to stop.")

    while True:

        user_input = input("\nYour question: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            print("Please enter a question.")
            continue

        if user_input.isdigit():

            index = int(user_input) - 1

            if 0 <= index < len(test_queries):

                question = test_queries[index]

            else:

                print("Invalid test query number.")
                continue

        else:

            question = user_input

        try:

            run_agent(agent, question)

        except Exception:

            logger.exception(
                "Agent execution failed."
            )

            print(
                "The agent encountered an error. "
                "Check the terminal logs."
            )


if __name__ == "__main__":
    main()
