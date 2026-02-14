"""
Configuration Management for Agentic RAG System
Centralized configuration with environment variable support.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """LLM Model configuration"""
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"
    max_reasoning_steps: int = 5
    max_tokens: int = 4096
    temperature: float = 0.0


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: Optional[int] = None  # Auto-detected


@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    default_top_k: int = 3
    max_top_k: int = 10
    similarity_threshold: float = 0.0


@dataclass
class VectorStoreConfig:
    """Vector database configuration"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "documents"


@dataclass
class ServerConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: str = "./uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB


@dataclass
class SystemConfig:
    """Complete system configuration"""
    model: ModelConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    vector_store: VectorStoreConfig
    server: ServerConfig


def load_config() -> SystemConfig:
    """
    Load configuration from environment variables with sensible defaults.
    
    Returns:
        SystemConfig: Complete system configuration
    """
    
    # Model configuration
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment. "
            "Please set it in your .env file or environment."
        )
    
    model_config = ModelConfig(
        anthropic_api_key=api_key,
        model_name=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_reasoning_steps=int(os.getenv("MAX_REASONING_STEPS", "5")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        temperature=float(os.getenv("TEMPERATURE", "0.0"))
    )
    
    # Embedding configuration
    embedding_config = EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Chunking configuration
    chunking_config = ChunkingConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    
    # Retrieval configuration
    retrieval_config = RetrievalConfig(
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "3")),
        max_top_k=int(os.getenv("MAX_TOP_K", "10")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    )
    
    # Vector store configuration
    vector_store_config = VectorStoreConfig(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "documents")
    )
    
    # Server configuration
    server_config = ServerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
        max_file_size=int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))
    )
    
    return SystemConfig(
        model=model_config,
        embedding=embedding_config,
        chunking=chunking_config,
        retrieval=retrieval_config,
        vector_store=vector_store_config,
        server=server_config
    )


def validate_config(config: SystemConfig) -> bool:
    """
    Validate configuration values.
    
    Args:
        config: System configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    
    # Validate API key
    if not config.model.anthropic_api_key:
        raise ValueError("Anthropic API key is required")
    
    # Validate chunking
    if config.chunking.chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    
    if config.chunking.chunk_overlap >= config.chunking.chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
    
    # Validate retrieval
    if config.retrieval.default_top_k < 1:
        raise ValueError("default_top_k must be positive")
    
    if config.retrieval.max_top_k < config.retrieval.default_top_k:
        raise ValueError("max_top_k must be >= default_top_k")
    
    # Validate server
    if not (1024 <= config.server.port <= 65535):
        raise ValueError("Server port must be between 1024 and 65535")
    
    return True


def print_config(config: SystemConfig):
    """Print configuration in a readable format"""
    
    print("\n" + "="*60)
    print("🔧 System Configuration")
    print("="*60)
    
    print("\n📊 Model Configuration:")
    print(f"  • Model: {config.model.model_name}")
    print(f"  • Max Reasoning Steps: {config.model.max_reasoning_steps}")
    print(f"  • Max Tokens: {config.model.max_tokens}")
    print(f"  • Temperature: {config.model.temperature}")
    
    print("\n🧮 Embedding Configuration:")
    print(f"  • Model: {config.embedding.model_name}")
    
    print("\n📄 Chunking Configuration:")
    print(f"  • Chunk Size: {config.chunking.chunk_size}")
    print(f"  • Chunk Overlap: {config.chunking.chunk_overlap}")
    
    print("\n🔍 Retrieval Configuration:")
    print(f"  • Default Top K: {config.retrieval.default_top_k}")
    print(f"  • Max Top K: {config.retrieval.max_top_k}")
    print(f"  • Similarity Threshold: {config.retrieval.similarity_threshold}")
    
    print("\n💾 Vector Store Configuration:")
    print(f"  • Persist Directory: {config.vector_store.persist_directory}")
    print(f"  • Collection Name: {config.vector_store.collection_name}")
    
    print("\n🌐 Server Configuration:")
    print(f"  • Host: {config.server.host}")
    print(f"  • Port: {config.server.port}")
    print(f"  • Upload Directory: {config.server.upload_dir}")
    print(f"  • Max File Size: {config.server.max_file_size / (1024*1024):.0f} MB")
    
    print("\n" + "="*60 + "\n")


# Example usage
if __name__ == "__main__":
    try:
        config = load_config()
        validate_config(config)
        print_config(config)
        print("✅ Configuration loaded and validated successfully!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
