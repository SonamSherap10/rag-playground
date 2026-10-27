from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

embedding = OllamaEmbeddings(model="nomic-embed-text")

document = '''# Authentication Guide

## 1. Overview

Our platform supports three ways to authenticate: username and password, OAuth 2.0 sign-in through a third-party provider, and API keys for server-to-server calls. Every request to a protected endpoint must carry proof of identity, either a session cookie or an Authorization header. Requests without valid proof receive a 401 response.

## 2. Password Authentication

Passwords must be at least 12 characters long. We never store passwords directly. Each password is hashed with bcrypt using a cost factor of 12, and a unique salt is generated for every user. If the hashing cost is raised in the future, existing hashes are upgraded the next time the user signs in.

To protect against brute-force attacks, an account is locked for 15 minutes after 5 failed login attempts within a 10-minute window. During the lockout, even the correct password is rejected and the API returns error AUTH-103. Administrators can unlock an account early from the admin console.

### Password Reset

Users who forgot their password can request a reset link by email. The link is single-use and expires after 30 minutes. After a successful reset, all existing sessions for that user are invalidated, so every device must sign in again. If the user did not request the reset, they should ignore the email and the link will expire on its own.

## 3. Tokens and Sessions

After a successful login, the server issues two tokens. The access token is a signed JWT that expires after 15 minutes and is sent in the Authorization header as a Bearer token. The refresh token lasts 7 days and is used only to obtain a new access token. Refresh tokens are rotated on every use, which means the old refresh token stops working as soon as a new one is issued.

To get a new access token, the client calls the refresh endpoint:'''

recursive_Chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap =50,
    separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
) 

recursive_chunks = recursive_Chunk_splitter.split_text(document)
# for i,chunk in enumerate(recursive_chunks):
#     print(f"\n ---Chunk :{i+1} Chunk lenght {len(chunk) } \n {chunk}")

Semantic_chunker = SemanticChunker(
    embedding,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)

Semantic_chunks = Semantic_chunker.split_text(document)
# for i, chunk in enumerate(Semantic_chunks):
#      print(f"\n ---Chunk :{i+1} Chunk lenght {len(chunk) } \n {chunk}")



recursive_vector_store = Chroma.from_texts(
     recursive_chunks,
     embedding,
     collection_name='recursive_chunks'
)

semantic_vector_store = Chroma.from_texts(
     Semantic_chunks,
     embedding,
     collection_name='semantic_chunks'
)


tests = [
    # (question, [every string that must appear in the retrieved chunks])
    ("What algorithm is used to hash passwords?", ["bcrypt"]),
    ("What error code is returned when an account is locked?", ["AUTH-103"]),
    ("How long is a password reset link valid and what happens to sessions after a reset?",
     ["30 minutes", "invalidated"]),
    ("How long do access tokens and refresh tokens last?", ["15 minutes", "7 days"]),
    ("Why can't I log in even with the right password after many tries?",
     ["failed login attempts"]),
]

#retrieval funcion 
def test_retrieval(query, must_contain, vectorstore, name, k=1):
    results = vectorstore.similarity_search(query, k=k)
    text = " ".join(d.page_content for d in results).lower()
    ok = all(s.lower() in text for s in must_contain)
    print(f"  {name:10} {'✅' if ok else '❌'}  top chunk: {results[0].page_content[:70]!r}...")
    return ok


print(f"\n{'=' * 60}\nRETRIEVAL TESTS\n{'=' * 60}")

scores = {"recursive": 0, "semantic": 0}
for query, must_contain in tests:
    print(f"\nQuery: {query}")
    scores["recursive"] += test_retrieval(query, must_contain, recursive_vector_store, "recursive")
    scores["semantic"] += test_retrieval(query, must_contain, semantic_vector_store, "semantic")

print(f"\n{'=' * 60}\nSCORES (k=1) out of {len(tests)}: {scores}")

for name, chunks in [("recursive", recursive_chunks), ("semantic", Semantic_chunks)]:
    sizes = [len(c) for c in chunks]
    print(f"{name}: {len(chunks)} chunks, min={min(sizes)}, max={max(sizes)}")