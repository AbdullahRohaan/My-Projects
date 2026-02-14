"""
FastAPI Server for Agentic RAG System (Groq Version)
Provides REST API endpoints for the system.
Modified to work with Groq with chat history.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
from pathlib import Path
import shutil

from agent_engine import AgenticRAGEngine, ConversationManager
from rag_components import RAGPipeline, VectorStore, TextChunker


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    confidence: float
    retrieved_contexts: List[Dict[str, Any]]
    reasoning_steps: List[str]
    tools_used: List[str]


class DocumentStats(BaseModel):
    total_documents: int
    collection_name: str
    embedding_dimension: int


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    total_chunks: int
    errors: int


class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    last_updated: float
    message_count: int
    preview: str


# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG System",
    description="Advanced AI-powered question-answering with autonomous decision-making and chat history",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⭐ PUT API KEY HERE ⭐
GROQ_API_KEY = "Api key here"

# Global instances
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = VectorStore(persist_directory="./chroma_db")
rag_pipeline = RAGPipeline(vector_store=vector_store)

# Initialize agent
agent_engine = AgenticRAGEngine(
    api_key=GROQ_API_KEY
) if GROQ_API_KEY and GROQ_API_KEY != "PASTE_YOUR_GROQ_API_KEY_HERE" else None

conversation_manager = ConversationManager(max_history=20)


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🚀 Agentic RAG System starting...")
    print(f"📊 Vector Store Stats: {vector_store.get_stats()}")
    
    if not agent_engine:
        print("⚠️  Warning: GROQ_API_KEY not set. Agent features will be disabled.")
        print("   Please set your API key in api_server.py (line 70)")
    else:
        print("✅ Groq API key detected! Agent is ready.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "system": "Agentic RAG",
        "agent_enabled": agent_engine is not None,
        "model": "llama-3.3-70b-versatile"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query using the Agentic RAG system.
    The agent autonomously decides when to retrieve and how to reason.
    """
    if not agent_engine:
        raise HTTPException(
            status_code=503,
            detail="Agent engine not available. Please set GROQ_API_KEY in api_server.py"
        )
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    conversation_history = conversation_manager.get_history(session_id, include_metadata=False)
    
    try:
        # Process query with agent
        answer, agent_state = await agent_engine.process_query(
            query=request.query,
            conversation_history=conversation_history,
            retrieval_function=lambda q, k: rag_pipeline.retrieve(q, k)
        )
        
        # Update conversation history
        conversation_manager.add_message(
            session_id, 
            "user", 
            request.query,
            metadata={"timestamp": agent_state.reasoning_steps}
        )
        conversation_manager.add_message(
            session_id, 
            "assistant", 
            answer,
            metadata={
                "confidence": agent_state.confidence,
                "tools_used": agent_state.tools_used
            }
        )
        
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            confidence=agent_state.confidence,
            retrieved_contexts=agent_state.retrieved_contexts,
            reasoning_steps=agent_state.reasoning_steps,
            tools_used=agent_state.tools_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=IngestResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the knowledge base.
    Supports: PDF, DOCX, TXT, MD, PNG, JPG, JPEG
    """
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check if it's an image file
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            # For images, just save them (OCR can be added later if needed)
            return IngestResponse(
                message=f"Image {file.filename} uploaded (OCR not yet implemented)",
                files_processed=1,
                total_chunks=0,
                errors=0
            )
        
        # Ingest the file
        chunks = rag_pipeline.ingest_file(str(file_path))
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            files_processed=1,
            total_chunks=chunks,
            errors=0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-multiple", response_model=IngestResponse)
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload and ingest multiple documents"""
    stats = {"files_processed": 0, "total_chunks": 0, "errors": 0}
    
    for file in files:
        try:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Skip images for now
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                stats["files_processed"] += 1
                continue
            
            chunks = rag_pipeline.ingest_file(str(file_path))
            stats["files_processed"] += 1
            stats["total_chunks"] += chunks
        
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            stats["errors"] += 1
    
    return IngestResponse(
        message=f"Processed {stats['files_processed']} files",
        **stats
    )


@app.get("/stats", response_model=DocumentStats)
async def get_stats():
    """Get statistics about the knowledge base"""
    stats = vector_store.get_stats()
    return DocumentStats(**stats)


@app.post("/search")
async def search_documents(query: str = Form(...), top_k: int = Form(3)):
    """Search for relevant documents without agent processing"""
    try:
        results = await rag_pipeline.retrieve(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all conversation sessions with metadata"""
    sessions = conversation_manager.list_sessions()
    return [SessionInfo(**session) for session in sessions]


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full conversation history for a session"""
    history = conversation_manager.get_history(session_id, include_metadata=True)
    info = conversation_manager.get_session_info(session_id)
    
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "info": info,
        "messages": history
    }


@app.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    conversation_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.delete("/clear-all-documents")
async def clear_all_documents():
    """Clear all documents from the knowledge base"""
    try:
        vector_store.delete_all()
        return {"message": "All documents cleared from knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-old-sessions")
async def clear_old_sessions(days: int = 7):
    """Clear sessions older than specified days"""
    conversation_manager.delete_old_sessions(days)
    return {"message": f"Cleared sessions older than {days} days"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "agent_engine": agent_engine is not None,
            "vector_store": True,
            "rag_pipeline": True
        },
        "stats": vector_store.get_stats(),
        "llm_provider": "Groq",
        "active_sessions": len(conversation_manager.conversations)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
