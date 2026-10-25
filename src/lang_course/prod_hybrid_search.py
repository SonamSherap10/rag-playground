from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever 

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text" )

documents = [
    Document(page_content="ERR-4012: Authentication token expired. The client must request a new access token by calling POST /v2/auth/refresh with the refresh token. Sessions older than 60 minutes are invalidated automatically.", metadata={"id": "doc1"}),
    Document(page_content="ERR-4013: Authentication failed because the API key was revoked by an administrator. Generate a new key in the developer console under Settings > API Keys.", metadata={"id": "doc2"}),
    Document(page_content="ERR-5031: Database connection pool exhausted. This happens when too many simultaneous requests compete for connections. Increase max_pool_size from 20 to 50 in config/db.yaml or add a read replica.", metadata={"id": "doc3"}),
    Document(page_content="Invoice INV-2024-0087 for customer Priya Sharma was refunded in full. The refund covers order ORD-88213, which contained one SKU-NB-2291 backpack that arrived damaged.", metadata={"id": "doc4"}),
    Document(page_content="SKU-NB-2291 is the Nimbus Backpack Pro. It is waterproof, holds 28 liters, and has a padded sleeve that fits laptops up to 16 inches. Designed for daily commuting and travel.", metadata={"id": "doc5"}),
    Document(page_content="SKU-NB-2292 is the Nimbus Backpack Lite. It is lightweight, holds 18 liters, and is water resistant but not waterproof. Best for short day trips and students.", metadata={"id": "doc6"}),
    Document(page_content="Refund policy: customers may return any item within 30 days of delivery. Once the item is inspected, the money is sent back to the original payment method within 5 to 7 business days.", metadata={"id": "doc7"}),
    Document(page_content="Ticket JIRA-7741: The worker service in release v3.2.1 had a memory leak that caused it to crash after several hours under load. Fixed in v3.2.2 by closing unused file handles.", metadata={"id": "doc8"}),
    Document(page_content="Deployment runbook: if a release causes failures in production, roll back immediately with `kubectl rollout undo deployment/payments-api -n prod`, then notify the on-call engineer in #incident-response.", metadata={"id": "doc9"}),
    Document(page_content="Slow queries on the orders table can be fixed by adding an index on orders.customer_id. Without it, the database scans every row, which makes dashboards and reports take much longer to load.", metadata={"id": "doc10"}),
    Document(page_content="If you forgot your password or cannot sign in, use the 'Forgot password' link on the login page. We will email you a one-time link that lets you choose a new password.", metadata={"id": "doc11"}),
    Document(page_content="Meeting notes 2026-03-14: Rajesh Karki proposed migrating the analytics store to PostgreSQL 16. The team agreed, with the cutover planned for the Q3 maintenance window.", metadata={"id": "doc12"}),
]

print(f"Number of documents: {len(documents)}")


#Vector retriever
embeddings= OllamaEmbeddings(model="nomic-embed-text" )

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="Hybrid_test"
)
vector_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector retriever initialized.")



#BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents=documents, k=3)
print("BM25 retriever initialized.")    

#ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retreiver, bm25_retriever],
      weights=[0.5, 0.5]  #Give equal weight to both retrievers. You can adjust the weights as needed to make it more semantic heavy or keyword-based.
      )
print("Ensemble retriever initialized.Hybrid search setup complete.")


def test_query(query,name,retreiver):
    results = retreiver.invoke(query)
    print(f"\nQuery: {query}")
    print(f"Retriever: {name}")
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:180] + '...'
        print(f'{i+1}.{preview}')
        return results

test_queries = [
    # --- Exact product code (BM25 should win) ---
    "SKU-NB-2291",
    "SKU-NB-2292",

    # --- Exact error code (BM25 should win; vectors may confuse 4012 vs 4013) ---
    "ERR-4012",
    "ERR-4013"

    # --- Exact ticket / invoice / version identifiers ---
    "JIRA-7741",
    "INV-2024-0087",
    "v3.2.1",

    # --- Acronym / technical shorthand ---
    "JWT refresh",
    "k8s rollback",
    "SKU meaning",

    # --- Semantic question (vectors should win, little keyword overlap) ---
    "my session timed out and I got logged out",
    "too many users hitting the database at once",
    "how do I get my money back for a bag",

    # --- General semantics (broad, conceptual) ---
    "which backpack is best for laptops and commuting",
    "lightweight bag for students",
    "how long does a refund take",
    "who suggested changing the analytics database",

    # --- Mixed: identifier + meaning (hybrid should win) ---
    "refund for INV-2024-0087",
    "which backpack is waterproof SKU-NB-2291",
    "ERR-4012 token"
    ]

for query in test_queries:
    print('=' * 60)

    #vector
    vector_result = test_query(query,"VECTOR",vector_retreiver)

    #BM25 
    bm25_result = test_query(query,"BM25",bm25_retriever)

    #Hybrid 
    hybrid_result= test_query(query,"Hybrid",ensemble_retriever)

