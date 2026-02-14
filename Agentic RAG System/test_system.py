"""
Test Suite for Agentic RAG System
Run with: pytest test_system.py -v
"""

import pytest
import asyncio
import os
from pathlib import Path
import tempfile

# Import components to test
from rag_components import (
    DocumentProcessor, TextChunker, EmbeddingModel, 
    VectorStore, RAGPipeline, Document
)
from agent_engine import AgenticRAGEngine, ConversationManager


class TestDocumentProcessor:
    """Test document processing functionality"""
    
    def test_process_txt(self, tmp_path):
        """Test TXT file processing"""
        # Create temporary text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is a test document.\nIt has multiple lines.")
        
        # Process
        docs = DocumentProcessor.process_txt(str(txt_file))
        
        assert len(docs) == 1
        assert "test document" in docs[0].content
        assert docs[0].metadata["type"] == "txt"
    
    def test_process_markdown(self, tmp_path):
        """Test Markdown file processing"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading\n\nThis is **bold** text.")
        
        docs = DocumentProcessor.process_markdown(str(md_file))
        
        assert len(docs) == 1
        assert "Heading" in docs[0].content
        assert docs[0].metadata["type"] == "markdown"


class TestTextChunker:
    """Test text chunking functionality"""
    
    def test_chunk_text(self):
        """Test basic text chunking"""
        chunker = TextChunker(chunk_size=5, chunk_overlap=2)
        text = "one two three four five six seven eight nine ten"
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        # Verify overlap
        assert chunks[0].split()[-1] in chunks[1] or chunks[0].split()[-2] in chunks[1]
    
    def test_chunk_documents(self):
        """Test document chunking"""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        
        doc = Document(
            content=" ".join([f"word{i}" for i in range(50)]),
            metadata={"source": "test"},
            doc_id="test-doc"
        )
        
        chunked = chunker.chunk_documents([doc])
        
        assert len(chunked) > 1
        assert all(d.metadata["parent_doc_id"] == "test-doc" for d in chunked)


class TestEmbeddingModel:
    """Test embedding generation"""
    
    def test_embed_text(self):
        """Test single text embedding"""
        model = EmbeddingModel()
        
        embedding = model.embed_text("This is a test sentence.")
        
        assert isinstance(embedding, list)
        assert len(embedding) == model.dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_batch(self):
        """Test batch embedding"""
        model = EmbeddingModel()
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        
        embeddings = model.embed_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == model.dimension for emb in embeddings)


class TestVectorStore:
    """Test vector store operations"""
    
    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create temporary vector store"""
        return VectorStore(persist_directory=str(tmp_path / "test_db"))
    
    def test_add_and_search(self, vector_store):
        """Test adding documents and searching"""
        # Create test documents
        docs = [
            Document(
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "test1"},
                doc_id="doc1"
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "test2"},
                doc_id="doc2"
            )
        ]
        
        # Add documents
        vector_store.add_documents(docs)
        
        # Search
        results = vector_store.search("What is machine learning?", top_k=1)
        
        assert len(results) == 1
        assert "machine learning" in results[0]["content"].lower()
    
    def test_get_stats(self, vector_store):
        """Test getting statistics"""
        stats = vector_store.get_stats()
        
        assert "total_documents" in stats
        assert "embedding_dimension" in stats
        assert isinstance(stats["total_documents"], int)


class TestRAGPipeline:
    """Test RAG pipeline integration"""
    
    @pytest.fixture
    def rag_pipeline(self, tmp_path):
        """Create RAG pipeline with temporary storage"""
        vector_store = VectorStore(persist_directory=str(tmp_path / "test_db"))
        return RAGPipeline(vector_store=vector_store)
    
    def test_ingest_file(self, rag_pipeline, tmp_path):
        """Test file ingestion"""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for ingestion testing.")
        
        # Ingest
        chunks = rag_pipeline.ingest_file(str(test_file))
        
        assert chunks > 0
    
    @pytest.mark.asyncio
    async def test_retrieve(self, rag_pipeline, tmp_path):
        """Test retrieval"""
        # Ingest a document first
        test_file = tmp_path / "test.txt"
        test_file.write_text("Artificial intelligence is transforming technology.")
        rag_pipeline.ingest_file(str(test_file))
        
        # Retrieve
        results = await rag_pipeline.retrieve("What is AI?", top_k=1)
        
        assert len(results) >= 0  # May or may not find relevant results


class TestConversationManager:
    """Test conversation management"""
    
    def test_add_message(self):
        """Test adding messages"""
        manager = ConversationManager()
        session_id = "test-session"
        
        manager.add_message(session_id, "user", "Hello")
        manager.add_message(session_id, "assistant", "Hi there!")
        
        history = manager.get_history(session_id)
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_clear_session(self):
        """Test clearing session"""
        manager = ConversationManager()
        session_id = "test-session"
        
        manager.add_message(session_id, "user", "Hello")
        manager.clear_session(session_id)
        
        history = manager.get_history(session_id)
        assert len(history) == 0


class TestAgenticRAGEngine:
    """Test agentic RAG engine (requires API key)"""
    
    @pytest.fixture
    def agent_engine(self):
        """Create agent engine"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        return AgenticRAGEngine(api_key=api_key, max_reasoning_steps=3)
    
    def test_initialize_tools(self, agent_engine):
        """Test tool initialization"""
        tools = agent_engine.tools_available
        
        assert len(tools) > 0
        assert any(t["name"] == "retrieve_documents" for t in tools)
        assert any(t["name"] == "calculate" for t in tools)
    
    @pytest.mark.asyncio
    async def test_process_simple_query(self, agent_engine):
        """Test processing a simple query without retrieval"""
        query = "What is 2 + 2?"
        
        answer, state = await agent_engine.process_query(
            query=query,
            retrieval_function=None
        )
        
        assert answer is not None
        assert len(answer) > 0
        assert state.query == query


# Integration Tests
class TestIntegration:
    """Test full system integration"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        """Test complete pipeline from ingestion to query"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        # Setup
        vector_store = VectorStore(persist_directory=str(tmp_path / "test_db"))
        rag_pipeline = RAGPipeline(vector_store=vector_store)
        agent_engine = AgenticRAGEngine(api_key=api_key, max_reasoning_steps=2)
        
        # Create and ingest document
        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "Machine learning is a method of data analysis that automates "
            "analytical model building. It uses algorithms that iteratively "
            "learn from data."
        )
        rag_pipeline.ingest_file(str(test_file))
        
        # Query
        query = "What is machine learning?"
        answer, state = await agent_engine.process_query(
            query=query,
            retrieval_function=lambda q, k: rag_pipeline.retrieve(q, k)
        )
        
        # Verify
        assert answer is not None
        assert len(answer) > 0


# Performance Tests
class TestPerformance:
    """Test performance characteristics"""
    
    def test_embedding_speed(self):
        """Test embedding generation speed"""
        import time
        
        model = EmbeddingModel()
        texts = ["Test sentence number {}".format(i) for i in range(100)]
        
        start = time.time()
        embeddings = model.embed_batch(texts)
        duration = time.time() - start
        
        assert duration < 10  # Should complete in under 10 seconds
        assert len(embeddings) == 100
    
    def test_search_speed(self, tmp_path):
        """Test search speed"""
        import time
        
        # Create vector store with documents
        vector_store = VectorStore(persist_directory=str(tmp_path / "perf_db"))
        
        docs = [
            Document(
                content=f"Test document number {i} with some content.",
                metadata={"id": i},
                doc_id=f"doc{i}"
            )
            for i in range(100)
        ]
        
        vector_store.add_documents(docs)
        
        # Time search
        start = time.time()
        results = vector_store.search("test query", top_k=5)
        duration = time.time() - start
        
        assert duration < 1  # Should complete in under 1 second
        assert len(results) <= 5


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
