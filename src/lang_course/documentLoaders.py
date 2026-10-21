import os 
import tempfile
from langchain_community.document_loaders import (TextLoader,PyPDFLoader)

from dotenv import load_dotenv
load_dotenv()

def load_test_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a test file for document loading.")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        print(documents)

        for doc in documents:
            print(f"Document loaded successfully.{doc}")
            print(f"Document content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")

    finally:
        os.remove(temp_file_path)   # Cean up the temporary file after loading     


if __name__ == "__main__":
    load_test_file()
    


def pdfLoader(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(documents)

    print(f"Document of length {len(documents)} loaded successfully.{documents}")

    for i, doc in enumerate(documents):
        print(f"Document {i+1} content: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    # Example usage of pdfLoader function
    pdf_path = "./docs/spm report.pdf"  
    pdfLoader(pdf_path)