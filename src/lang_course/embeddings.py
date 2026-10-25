from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import numpy as np
load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text" )

def basic_embeddings():
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimension: {len(single_embedding)}")
    print(f"First five values of the embedding vector: {single_embedding[:5]}")
    print(f"Normalized embedding vector: { np.linalg.norm(single_embedding)}")

def batch_embeddings():
    texts= [
        "What is Machine Learning?",
        "What is Artificial Intelligence?",
        "What is Deep Learning?"
    ]
    batch_embedding = embeddings.embed_documents(texts)
    for i, emb in enumerate(batch_embedding):
        print(f"Text: {texts[i]}")
        print(f"Vector dimension: {len(emb)}")
        print(f"First five values of the embedding vector: {emb[:5]}")
        print(f"Normalized embedding vector: { np.linalg.norm(emb)}")
        print("")


def similarity_search():
    docs=[
        "python is a programming language",
        "javaScript is used for web development",
        "java is a object oriented language used for backend development",
        "Machine learning is a subset of artificial intelligence",
        "I love both dogs and cats"
    ]
    query = "What programming languages exist?"

    doc_embeddings = embeddings.embed_documents(docs)
    query_embedding = embeddings.embed_query(query)

    def cosine_similarity(vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))   

    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    #rank the documents based on similarity scores
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    print("Ranked documents based on similarity to the query:")
    for doc, score in ranked_docs:
        print(f"Document: {doc}, Similarity Score: {score}")


if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()