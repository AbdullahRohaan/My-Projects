"""
Example usage script for the Agentic RAG System
Demonstrates how to use the system programmatically.
"""

import asyncio
import os
from agent_engine import AgenticRAGEngine, ConversationManager
from rag_components import RAGPipeline, VectorStore, TextChunker


async def main():
    """
    Demonstrates programmatic usage of the Agentic RAG system.
    """
    print("🤖 Agentic RAG System - Example Usage\n")
    print("=" * 60)
    
    # Initialize components
    print("\n1️⃣ Initializing components...")
    
    # Set your API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("   Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Create vector store
    vector_store = VectorStore(persist_directory="./example_db")
    
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        chunker=TextChunker(chunk_size=500, chunk_overlap=50)
    )
    
    # Create agent engine
    agent = AgenticRAGEngine(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        max_reasoning_steps=5
    )
    
    # Create conversation manager
    conversation_manager = ConversationManager()
    
    print("✅ Components initialized!")
    
    # Ingest sample documents
    print("\n2️⃣ Ingesting sample documents...")
    
    sample_docs_dir = "./sample_docs"
    if os.path.exists(sample_docs_dir):
        stats = rag_pipeline.ingest_directory(sample_docs_dir)
        print(f"✅ Ingested {stats['files_processed']} files")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Errors: {stats['errors']}")
    else:
        print("⚠️  Sample documents directory not found")
        print("   Continuing without document ingestion...")
    
    # Display vector store stats
    print("\n3️⃣ Vector Store Statistics:")
    stats = vector_store.get_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    
    # Example queries
    print("\n4️⃣ Running example queries...\n")
    print("=" * 60)
    
    session_id = "example-session"
    
    queries = [
        {
            "query": "What is machine learning?",
            "description": "Simple factual question"
        },
        {
            "query": "Compare supervised and unsupervised learning approaches.",
            "description": "Comparison requiring retrieval"
        },
        {
            "query": "If I have 10000 samples and use 80-20 split, how many training samples?",
            "description": "Mathematical calculation"
        },
        {
            "query": "What are the main challenges in RAG systems according to the documents?",
            "description": "Document-grounded query"
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        query = query_info["query"]
        description = query_info["description"]
        
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {description}")
        print(f"{'─' * 60}")
        print(f"Q: {query}\n")
        
        # Get conversation history
        history = conversation_manager.get_history(session_id)
        
        try:
            # Process query
            answer, agent_state = await agent.process_query(
                query=query,
                conversation_history=history,
                retrieval_function=lambda q, k: rag_pipeline.retrieve(q, k)
            )
            
            # Update conversation
            conversation_manager.add_message(session_id, "user", query)
            conversation_manager.add_message(session_id, "assistant", answer)
            
            # Display results
            print(f"A: {answer}\n")
            
            print("📊 Agent Metadata:")
            print(f"   • Confidence: {agent_state.confidence:.2%}")
            print(f"   • Tools Used: {', '.join(agent_state.tools_used) or 'None'}")
            print(f"   • Reasoning Steps: {len(agent_state.reasoning_steps)}")
            print(f"   • Retrieved Contexts: {len(agent_state.retrieved_contexts)}")
            
            if agent_state.retrieved_contexts:
                print("\n📚 Retrieved Context Preview:")
                for idx, ctx in enumerate(agent_state.retrieved_contexts[:2], 1):
                    preview = ctx['content'][:100].replace('\n', ' ')
                    source = ctx.get('metadata', {}).get('source', 'Unknown')
                    print(f"   {idx}. [{source}] {preview}...")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("\nKey Observations:")
    print("  • The agent decided when to retrieve documents")
    print("  • Different tools were used for different query types")
    print("  • Confidence scores varied based on available information")
    print("  • Multi-step reasoning was employed when needed")
    
    print("\n🎯 Next Steps:")
    print("  1. Run 'python api_server.py' to start the web server")
    print("  2. Open 'index.html' in your browser")
    print("  3. Upload your own documents and start querying!")
    
    # Cleanup option
    print("\n🧹 Cleanup:")
    cleanup = input("Do you want to clear the example database? (y/n): ")
    if cleanup.lower() == 'y':
        vector_store.delete_all()
        print("✅ Database cleared!")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         🤖 Agentic RAG System - Example Usage           ║
    ║                                                           ║
    ║  This script demonstrates programmatic usage of the       ║
    ║  Agentic RAG system with autonomous decision-making.      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
