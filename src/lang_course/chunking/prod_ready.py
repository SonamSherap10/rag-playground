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


def smart_chunker(text:str,use_semantic:bool = True,fallback_chunk_size :int = 500):
    if(use_semantic):
        try:
            chunker =SemanticChunker(
                embedding,
                breakpoint_threshold_amount=90,
                breakpoint_threshold_type='percentile'
            )
            chunks= chunker.split_text(document)
            
            max_chunk_size = 2000
            if any(len(c)>max_chunk_size for c in chunks):
                return _recursive_fallback(text,fallback_chunk_size)
            return chunks
        
        except Exception as e:
            print(f"Semantic chunking failed{e}")
            return _recursive_fallback(text,fallback_chunk_size)
    
    return _recursive_fallback(text,fallback_chunk_size)

def _recursive_fallback(text:str,chunk_size:int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= 50
    )
    return splitter.split_text(text)

chunks= smart_chunker(document,use_semantic=True)
print(f"'Created {len(chunks)} semantic chunks")
