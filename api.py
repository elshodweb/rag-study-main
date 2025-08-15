#!/usr/bin/env python3
"""
Simple RAG API Server - Everything in one file
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime
import uuid
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

prompt = ChatPromptTemplate.from_template(
    """
    Вы - полезный помощник, который отвечает на вопросы на основе предоставленного контекста.
    Используйте следующий контекст для ответа на вопрос. Если вы не знаете ответ, просто скажите, что не знаете.
    
    Вопрос: {question}
    Контекст: {context}
    
    Ответ:
    """
)

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

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple RAG API",
        "endpoints": {
            "save_document": "/save-document",
            "ask_question": "/ask-question"
        }
    }

@app.post("/save-document", response_model=DocumentResponse)
async def save_document(doc_input: DocumentInput):
    """Save a new document"""
    try:
        # Generate unique ID
        doc_id = doc_input.doc_id
        
        # Create document
        doc = Document(
            page_content=doc_input.content,
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
            content=doc_input.content[:200] + "..." if len(doc_input.content) > 200 else doc_input.content,
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
        
        # Combine document content
        doc_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
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
    print("🚀 Starting Simple RAG API Server...")
    print("📚 Available endpoints:")
    print("   POST /save-document - Save documents")
    print("   POST /ask-question - Ask questions")
    print(f"📖 API Docs: http://localhost:{PORT}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=True
    )
