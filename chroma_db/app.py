import chromadb

chroma_client = chromadb.Client()

collection_name = "test"
collection = chroma_client.get_or_create_collection(
    name=collection_name
)

documents = [
    {"id": "doc1", "text": "Hello to the planet"},
    {"id": "doc2", "text": "This is the second document."},
    {"id": "doc3", "text": "This is the third document."}
]

for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]]
    )

query_text = "Hello, world"

results = collection.query(
    query_texts=[query_text],
    n_results=3
)

print(results)