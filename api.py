#!/usr/bin/env python3
"""
Simple RAG API Server - Everything in one file
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Optional HTML -> Markdown converter
try:
    from markdownify import markdownify as _html_to_md
except Exception:
    _html_to_md = None

# Configuration from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Initialize FastAPI
app = FastAPI(
    title="Simple RAG API",
    description="API для сохранения документов и ответов на вопросы",
    version="1.0.0"
)

# Initialize RAG components
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_DIR,
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY
)


def convert_html_to_markdown_if_needed(content: str) -> str:
    """Convert HTML content to Markdown when input looks like HTML.

    Uses markdownify when available; falls back to BeautifulSoup text extraction.
    Non-HTML content is returned unchanged.
    """
    if not content:
        return content

    # Heuristic: detect HTML tags
    import re
    looks_like_html = bool(re.search(r"<[^>]+>", content))
    if not looks_like_html:
        return content

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    cleaned_html = str(soup)

    if _html_to_md is not None:
        try:
            markdown_text = _html_to_md(cleaned_html)
        except Exception:
            markdown_text = soup.get_text(separator="\n", strip=True)
    else:
        markdown_text = soup.get_text(separator="\n", strip=True)

    # Normalize spacing
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text).strip()
    return markdown_text


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Pydantic models
class DocumentInput(BaseModel):
    content: str
    doc_id: str
    name: Optional[str] = None
    organization: Optional[str] = None

class DocumentResponse(BaseModel):
    doc_id: str
    name: str
    content: str
    organization: str

class QuestionInput(BaseModel):
    question: str
    doc_id: Optional[str] = None
    name: Optional[str] = None
    organization: Optional[str] = None

class QuestionResponse(BaseModel):
    question: str
    answer: str
    documents_used: List[Dict[str, str]]


@app.post("/save-document", response_model=DocumentResponse)
async def save_document(doc_input: DocumentInput):
    """Save a new document"""
    try:
        # Generate unique ID
        doc_id = doc_input.doc_id
        
        # Normalize content: convert HTML to Markdown when needed
        normalized_content = convert_html_to_markdown_if_needed(doc_input.content)
        # Create document
        doc = Document(
            page_content=normalized_content,
            metadata={
                "doc_id": doc_id,
                "name": doc_input.name or f"Document {doc_id}",
                "organization": doc_input.organization or "Default Organization",
                "created_at": datetime.now().isoformat()
            }
        )
                
        # Split document into chunks
        splits = text_splitter.split_documents([doc])
        
        # Add to vector store
        ids = vector_store.add_documents(splits)
        return DocumentResponse(
            doc_id=doc_id,
            name=doc.metadata["name"],
            content=normalized_content[:200] + "..." if len(normalized_content) > 200 else normalized_content,
            organization=doc.metadata["organization"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении документа: {str(e)}")

@app.post("/ask-question", response_model=QuestionResponse)
async def ask_question(question_input: QuestionInput):
    """Ask a question and get answer"""
    try:
        # Build filters based on input (use only one filter at a time)
        filters = {}
        if question_input.doc_id:
            filters["doc_id"] = question_input.doc_id
        elif question_input.name:
            filters["name"] = question_input.name
        elif question_input.organization:
            filters["organization"] = question_input.organization

        # Search documents with filters
        if filters:
            retrieved_docs = vector_store.similarity_search(
                question_input.question,
                k=3,
                filter=filters
            )
        else:
            retrieved_docs = vector_store.similarity_search(question_input.question, k=3)
        
        if not retrieved_docs:
            return QuestionResponse(
                question=question_input.question,
                answer="Извините, не удалось найти релевантные документы для ответа на ваш вопрос.",
                documents_used=[],
            )
        
        # Combine document content with metadata
        doc_content_parts = []
        for doc in retrieved_docs:
            doc_info = f"name:{doc.metadata.get('name', 'N/A')}, doc_id: {doc.metadata.get('doc_id', 'N/A')},\nСодержание:\n{doc.page_content}"
            doc_content_parts.append(doc_info)
        
        doc_content = "\n\n---\n\n".join(doc_content_parts)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant. Answer only based on the provided context.
        If no information is found, write: "information not found" in the question language.
        Rules:
        If an answer is found, give a clear and complete explanation with quotes from the context.
        If the answer contains HTML — remove all tags, use Markdown.
        At the end of the answer, add a list of used documents in link format, template: [name](FRONTEND_URL.com/doc_id).
        
        Answer language = question language:
            Uzbek question → answer in Uzbek.
            Russian question → answer in Russian.

        Question: {question}
        Context: {context}
        Answer:
        """)
        
        # Generate answer
        messages = prompt.invoke({
            "question": question_input.question,
            "context": doc_content
        })
        answer = llm.invoke(messages)
        
        # Prepare documents info (group by doc_id to show unique documents)
        documents_info = []
        seen_doc_ids = set()
        
        for doc in retrieved_docs:
            doc_id = doc.metadata.get("doc_id", "N/A")
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                documents_info.append({
                    "doc_id": doc_id,
                    "name": doc.metadata.get("name", "N/A"),
                    "organization": doc.metadata.get("organization", "N/A")
                })
        
        return QuestionResponse(
            question=question_input.question,
            answer=answer.content,
            documents_used=documents_info,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}")

if __name__ == "__main__":
    print(f"📖 API Docs: http://localhost:{PORT}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=True
    )
