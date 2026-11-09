
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. SAMPLE DOCUMENT
# --------------------------------------------------

document_text = """
Project Orion is a billing management platform used by businesses
to manage customer subscriptions and payments.

The platform uses PostgreSQL as its primary database.
It stores customer information, subscription details,
payment records, and billing history.

The system automatically creates a database backup
every six hours. These backups are stored in encrypted
storage to protect customer information.

For authentication, Project Orion uses JWT access tokens.
Access tokens expire after 15 minutes. Refresh tokens
are used to obtain new access tokens without requiring
the user to log in again.

The billing service communicates with the payment service
through internal REST APIs. If a payment fails, the billing
service retries the transaction up to three times.
"""


# --------------------------------------------------
# 2. INITIALIZE THE LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# --------------------------------------------------
# 3. SPLIT THE DOCUMENT INTO NORMAL CHUNKS
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50
)

original_chunks = splitter.create_documents(
    [document_text]
)


# --------------------------------------------------
# 4. CREATE THE CONTEXT GENERATION PROMPT
# --------------------------------------------------

context_prompt = ChatPromptTemplate.from_template("""
You are helping prepare document chunks for a RAG system.

Below is the complete original document:

{document}

Below is one chunk extracted from that document:

{chunk}

Write a short context description explaining:

1. What this chunk is about.
2. Which project, system, or topic it belongs to.
3. Any important relationship with the surrounding document.

Rules:
- Write only the context description.
- Do not repeat the chunk itself.
- Do not invent information.
- Keep it concise, around 1-3 sentences.
""")


# --------------------------------------------------
# 5. GENERATE CONTEXT FOR EACH CHUNK
# --------------------------------------------------

contextualized_chunks = []

for index, chunk in enumerate(original_chunks):

    print(f"\nGenerating context for chunk {index + 1}...")

    response = llm.invoke(
        context_prompt.format_messages(
            document=document_text,
            chunk=chunk.page_content
        )
    )

    context = response.content.strip()

    # Combine generated context with original chunk.
    enriched_content = (
        f"Context: {context}\n\n"
        f"{chunk.page_content}"
    )

    contextualized_chunks.append(
        Document(
            page_content=enriched_content,
            metadata={
                "chunk_id": index,
                "chunk_type": "contextual"
            }
        )
    )


# --------------------------------------------------
# 6. DISPLAY THE RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONTEXTUAL CHUNKING RESULTS")
print("=" * 60)

for chunk in contextualized_chunks:

    print(f"\nChunk ID: {chunk.metadata['chunk_id']}")
    print("-" * 60)
    print(chunk.page_content)

print("\nTotal original chunks:", len(original_chunks))
print("Total contextualized chunks:", len(contextualized_chunks))
