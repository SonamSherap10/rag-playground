from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted reciprocal rank fusion."""
    doc_scores = {}
    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + 1 + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in sorted_docs[:k]]

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


class HybridRetreiver:
    "hybrid retreiver with bm25 and vector search"
    def __init__(self,documents:List[Document],bm25_weight : float=0.5,k :int=4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1-bm25_weight

        #initialize embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        #create vector store and retreiver
        self.vector_store = Chroma.from_documents(documents,self.embeddings,collection_name="hybrid_search")
        self.vector_retreiver = self.vector_store.as_retriever(search_kwargs={"k":k})
        print("Vector retriever initialized")

        #create bm25 retreiver
        self.bm25_retriever = BM25Retriever.from_documents(documents,k=k)
        print("BM25 retriever initialized")


    def search(self,query = str) -> List[Document]:
        "Running hybrid search using weighted rrf"
        return hybrid_retrieve(
            query,
            retrievers=[self.vector_retreiver,self.bm25_retriever],
            weights=[self.vector_weight,self.bm25_weight],
            k=self.k
        )

    def add_document(self,documents:List[Document]):
        "Add new documents to both retrievers"
        #Add to vector store
        self.vector_store.add_documents(documents)

        #recretae bm25 (as it dosent support incremental adds)
        all_docs = self.vector_store.get()
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=doc)for doc in all_docs["documnets"]],k=self.k
        )

#usage
retriever = HybridRetreiver(documents,bm25_weight=0.5,k=4)
results = retriever.search("ORD-88213 details")

for doc in results:
    print(doc.metadata["id"], doc.page_content[:80])