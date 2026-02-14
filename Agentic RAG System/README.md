# Agentic RAG System

An advanced AI-powered question-answering platform that combines Retrieval-Augmented Generation (RAG) with LLM-based autonomous agents.

## 🌟 Key Features

### Autonomous Decision-Making
Unlike traditional RAG systems that follow a fixed pipeline (retrieve → generate), this system introduces an **agentic decision layer** where the LLM dynamically decides:

- **When to retrieve information** - Only fetches documents when genuinely needed
- **Which tools to call** - Selects appropriate tools (retrieval, calculation, reasoning)
- **How many reasoning steps are required** - Breaks down complex queries into multiple steps
- **Whether the available information is sufficient** - Assesses confidence and asks for clarification

### Advanced Capabilities

✅ **Multi-step Reasoning**: Handles complex queries requiring logical decomposition  
✅ **Dynamic Tool Selection**: Chooses the right tools for each task  
✅ **Confidence Assessment**: Evaluates answer quality and reliability  
✅ **Context-Aware Retrieval**: Retrieves only when necessary to reduce hallucinations  
✅ **Conversation Memory**: Maintains session state across multiple interactions  
✅ **Multiple Document Formats**: Supports PDF, DOCX, TXT, and Markdown  

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Web)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI Server                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Agentic RAG Engine (Core)                  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Decision Layer (Claude Sonnet 4)              │  │   │
│  │  │  • When to retrieve?                           │  │   │
│  │  │  • Which tools to use?                         │  │   │
│  │  │  • Multi-step reasoning                        │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │  Retrieval   │  │ Calculator   │  │  Reasoning │  │   │
│  │  │     Tool     │  │     Tool     │  │    Tool    │  │   │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   RAG Components                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Document Processing                                  │   │
│  │  • PDF, DOCX, TXT, MD parsing                        │   │
│  │  • Intelligent chunking with overlap                 │   │
│  │  • Metadata extraction                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Embedding & Vector Store (ChromaDB)                 │   │
│  │  • Sentence-BERT embeddings                          │   │
│  │  • Semantic similarity search                        │   │
│  │  • Persistent storage                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.9 or higher
- Anthropic API Key (for Claude)
- 4GB+ RAM recommended
- Modern web browser

## 🚀 Installation

### 1. Clone or Create Project Directory

```bash
mkdir agentic-rag-system
cd agentic-rag-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **FastAPI & Uvicorn**: Web server and API framework
- **Anthropic SDK**: Claude AI integration
- **ChromaDB**: Vector database for document storage
- **Sentence Transformers**: Embedding generation
- **Document Processors**: PyPDF, python-docx, openpyxl
- Supporting libraries for data processing

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Get your API key**: [https://console.anthropic.com/](https://console.anthropic.com/)

## 💻 Usage

### Starting the Server

```bash
python api_server.py
```

The server will start on `http://localhost:8000`

You should see:
```
🚀 Agentic RAG System starting...
📊 Vector Store Stats: {'total_documents': 0, ...}
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Opening the Web Interface

1. Open your browser
2. Navigate to `http://localhost:8000` (for API docs)
3. Or open `index.html` directly in your browser

### Using the System

#### 1. Upload Documents

- Click the upload area or drag & drop files
- Supported formats: PDF, DOCX, TXT, MD
- Multiple files can be uploaded at once
- Documents are automatically processed and indexed

#### 2. Ask Questions

Type your question in the chat interface. The agent will:

1. **Analyze** the query to understand intent
2. **Decide** whether retrieval is needed
3. **Retrieve** relevant documents if necessary
4. **Reason** through multiple steps for complex queries
5. **Generate** a grounded, confident answer

#### 3. View Agent Reasoning

Each response includes:
- **Confidence Score**: Agent's certainty about the answer
- **Tools Used**: Which capabilities were leveraged
- **Reasoning Steps**: How many thinking iterations occurred
- **Retrieved Contexts**: Source documents used (if any)

## 🔧 API Endpoints

### POST `/query`
Process a question with the agentic system.

```json
{
  "query": "What is machine learning?",
  "session_id": "optional-session-id",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "session_id": "uuid",
  "confidence": 0.85,
  "retrieved_contexts": [...],
  "reasoning_steps": ["Step 1", "Step 2"],
  "tools_used": ["retrieve_documents"]
}
```

### POST `/upload`
Upload a single document.

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### POST `/upload-multiple`
Upload multiple documents.

### GET `/stats`
Get knowledge base statistics.

### POST `/search`
Direct document search without agent processing.

### DELETE `/clear-all-documents`
Clear the entire knowledge base.

## 📝 Example Queries

### Simple Factual Questions
```
Q: What is artificial intelligence?
A: [Agent directly answers without retrieval if confident]
```

### Complex Multi-Step Queries
```
Q: Compare supervised and unsupervised learning, then explain which 
   would be better for customer segmentation.
A: [Agent breaks down into steps, retrieves relevant info, synthesizes]
```

### Calculation Queries
```
Q: If a dataset has 10,000 samples and I use 80-20 train-test split, 
   how many samples for training?
A: [Agent uses calculator tool: 10000 * 0.8 = 8000]
```

### Document-Grounded Questions
```
Q: According to the documents, what are the main challenges in RAG systems?
A: [Agent retrieves relevant sections and synthesizes answer with citations]
```

## 🧠 How Agentic RAG Works

### Traditional RAG Flow
```
User Query → Retrieve Documents → Generate Answer
```
**Problem**: Always retrieves, even when unnecessary

### Agentic RAG Flow
```
User Query → Agent Decides → [Maybe Retrieve] → [Maybe Calculate] 
          → [Maybe Multi-Step Reason] → Generate Answer
```
**Advantage**: Dynamic, intelligent decision-making

### Decision Process

The agent uses Claude Sonnet 4 with tool-calling capabilities:

1. **Assessment**: Analyze if the query needs external information
2. **Planning**: Decide which tools to use and in what order
3. **Execution**: Call tools (retrieve, calculate, reason)
4. **Synthesis**: Combine information and generate response
5. **Validation**: Assess confidence and completeness

## 🎯 Advanced Features

### Multi-Step Reasoning

For complex queries, the agent can:
- Break down the problem into sub-questions
- Solve each step sequentially
- Combine results into a coherent answer

### Confidence Assessment

The system evaluates answer quality based on:
- Number of retrieved documents
- Reasoning depth
- Presence of uncertainty phrases
- Information completeness

### Session Management

Maintains conversation context:
- Previous questions and answers
- User preferences
- Clarification history

## 🛠️ Customization

### Adjust Chunking Strategy

Edit `rag_components.py`:
```python
chunker = TextChunker(
    chunk_size=500,  # Adjust chunk size
    chunk_overlap=50  # Adjust overlap
)
```

### Change Embedding Model

Edit `rag_components.py`:
```python
embedding_model = EmbeddingModel(
    model_name="all-MiniLM-L6-v2"  # Or another model
)
```

### Modify Agent Behavior

Edit `agent_engine.py` system prompt to change:
- Decision-making criteria
- Tool usage preferences
- Response style

### Add Custom Tools

In `agent_engine.py`, add to `_initialize_tools()`:
```python
{
    "name": "custom_tool",
    "description": "What it does",
    "input_schema": {...}
}
```

Then implement in `_execute_tool()`.

## 📊 Sample Documents

The system includes three example documents:

1. **ai_overview.md**: Introduction to AI and its history
2. **machine_learning_guide.md**: Comprehensive ML guide
3. **rag_deep_dive.md**: Deep dive into RAG systems

To test with these:
```bash
# The documents are in sample_docs/
# Upload them through the web interface
```

## 🔍 Troubleshooting

### "Agent engine not available"
- Ensure `ANTHROPIC_API_KEY` is set in environment
- Restart the server after setting the key

### "No documents found"
- Upload documents first through `/upload` endpoint
- Check that files are in supported formats

### Slow Response Times
- Reduce `top_k` parameter (fewer documents retrieved)
- Use smaller documents or better chunking
- Consider using a local LLM for faster inference

### High Memory Usage
- Reduce chunk size
- Clear old documents periodically
- Use a smaller embedding model

## 📚 Further Reading

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Research Papers](https://arxiv.org/abs/2005.11401)

## 🤝 Contributing

This is a demonstration project. Feel free to:
- Extend with new tools
- Add support for more document types
- Improve the UI/UX
- Optimize performance

## 📄 License

MIT License - Use freely for educational and commercial purposes.

## 🎓 Learning Resources

### Understanding the Code

1. **agent_engine.py**: Core agentic decision-making logic
2. **rag_components.py**: Document processing and retrieval
3. **api_server.py**: REST API and orchestration
4. **index.html**: Web interface

### Key Concepts

- **Tool Calling**: How LLMs decide which tools to use
- **Vector Embeddings**: Converting text to numerical representations
- **Semantic Search**: Finding similar documents
- **Agentic Behavior**: Autonomous decision-making in AI

## 🚧 Future Enhancements

Potential improvements:
- [ ] Support for images and PDFs with OCR
- [ ] Multi-modal retrieval (text + images)
- [ ] Fine-tuned reranking models
- [ ] Support for multiple vector databases
- [ ] Graph-based RAG for knowledge graphs
- [ ] Streaming responses
- [ ] User authentication
- [ ] Analytics dashboard

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Examine server logs for errors
4. Ensure all dependencies are installed correctly

---

**Built with ❤️ using Claude Sonnet 4, ChromaDB, and FastAPI**

Happy querying! 🎉
