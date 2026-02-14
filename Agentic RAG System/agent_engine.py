import json
from groq import Groq
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class AgentAction(Enum):
    """Possible actions the agent can take"""
    RETRIEVE_DOCUMENTS = "retrieve_documents"
    CALCULATE = "calculate"
    GENERATE_ANSWER = "generate_answer"
    ASK_CLARIFICATION = "ask_clarification"
    MULTI_STEP_REASONING = "multi_step_reasoning"


@dataclass
class AgentState:
    """Tracks the current state of the agent's reasoning process"""
    query: str
    conversation_history: List[Dict[str, str]]
    retrieved_contexts: List[Dict[str, Any]]
    reasoning_steps: List[str]
    tools_used: List[str]
    confidence: float
    needs_more_info: bool


class AgenticRAGEngine:
    """
    Core engine that enables autonomous decision-making for RAG operations.
    The agent decides when to retrieve, what tools to use, and how to reason.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_reasoning_steps: int = 5
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_reasoning_steps = max_reasoning_steps
        self.tools_available = self._initialize_tools()
        
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Define tools available to the agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_documents",
                    "description": "Retrieves relevant documents from the vector database based on semantic similarity to the query. Use this when you need factual information from the knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant documents"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of documents to retrieve (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Performs mathematical calculations. Use when the query requires numerical computation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "multi_step_reasoning",
                    "description": "Break down a complex query into multiple reasoning steps. Use for queries that require logical decomposition.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reasoning steps to follow"
                            }
                        },
                        "required": ["steps"]
                    }
                }
            }
        ]
    
    def _create_system_prompt(self) -> str:
        """Creates the system prompt that guides agent behavior"""
        return """You are an autonomous RAG agent with the ability to make decisions about information retrieval and reasoning.

Your capabilities:
1. You can decide WHEN to retrieve information from the knowledge base
2. You can choose WHICH tools to use and in what order
3. You can perform MULTI-STEP REASONING for complex queries
4. You can assess whether you have SUFFICIENT INFORMATION to answer

Decision-making principles:
- Only retrieve documents when you genuinely need external information
- If you can answer from your training data reliably, do so without retrieval
- For ambiguous queries, break them down into steps
- Always assess your confidence level
- If information is contradictory or insufficient, acknowledge it

You have access to tools. Use them strategically, not reflexively."""

    async def process_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None,
        retrieval_function: Optional[callable] = None
    ) -> Tuple[str, AgentState]:
        """
        Process a query with autonomous decision-making.
        
        Args:
            query: User's question
            conversation_history: Previous conversation context
            retrieval_function: Function to retrieve documents from vector DB
            
        Returns:
            Tuple of (answer, agent_state)
        """
        if conversation_history is None:
            conversation_history = []
            
        agent_state = AgentState(
            query=query,
            conversation_history=conversation_history,
            retrieved_contexts=[],
            reasoning_steps=[],
            tools_used=[],
            confidence=0.0,
            needs_more_info=False
        )
        
        messages = [{"role": "system", "content": self._create_system_prompt()}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        
        reasoning_step = 0
        
        while reasoning_step < self.max_reasoning_steps:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_available,
                tool_choice="auto",
                max_tokens=4096
            )
            
            response_message = response.choices[0].message
            
            # Track reasoning step
            agent_state.reasoning_steps.append(f"Step {reasoning_step + 1}")
            
            # Check if tool calls were made
            if response_message.tool_calls:
                # Add assistant response to messages
                messages.append(response_message)
                
                # Process each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    agent_state.tools_used.append(function_name)
                    
                    # Execute the tool
                    tool_result = await self._execute_tool(
                        function_name,
                        function_args,
                        retrieval_function,
                        agent_state
                    )
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })
                
                reasoning_step += 1
            else:
                # No tool calls - we have the final answer
                final_answer = response_message.content
                
                # Assess confidence
                agent_state.confidence = self._assess_confidence(
                    final_answer,
                    agent_state
                )
                
                return final_answer, agent_state
        
        # Max steps reached
        return "I've reached my reasoning limit. The query may be too complex or require more information.", agent_state
    
    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        retrieval_function: Optional[callable],
        agent_state: AgentState
    ) -> Dict[str, Any]:
        """Execute a tool and return results"""
        
        if tool_name == "retrieve_documents":
            if retrieval_function is None:
                return {"error": "Retrieval function not provided"}
            
            query = tool_input.get("query")
            top_k = tool_input.get("top_k", 3)
            
            documents = await retrieval_function(query, top_k)
            agent_state.retrieved_contexts.extend(documents)
            
            return {
                "documents": [
                    {
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "relevance_score": doc.get("score", 0.0)
                    }
                    for doc in documents
                ]
            }
        
        elif tool_name == "calculate":
            expression = tool_input.get("expression")
            try:
                # Safe evaluation of mathematical expressions
                result = eval(expression, {"__builtins__": {}}, {})
                return {"result": result, "expression": expression}
            except Exception as e:
                return {"error": f"Calculation failed: {str(e)}"}
        
        elif tool_name == "multi_step_reasoning":
            steps = tool_input.get("steps", [])
            return {
                "reasoning_plan": steps,
                "status": "Plan created. Execute each step."
            }
        
        return {"error": f"Unknown tool: {tool_name}"}
    
    def _assess_confidence(self, answer: str, agent_state: AgentState) -> float:
        """
        Assess confidence based on various factors.
        This is a simplified heuristic - in production, use more sophisticated methods.
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence if documents were retrieved
        if len(agent_state.retrieved_contexts) > 0:
            confidence += 0.2
        
        # Increase confidence if multiple reasoning steps were used
        if len(agent_state.reasoning_steps) > 1:
            confidence += 0.1
        
        # Decrease confidence if answer contains uncertainty phrases
        uncertainty_phrases = ["might", "possibly", "unclear", "not sure", "may be"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))


class ConversationManager:
    """Manages conversation state and history with persistence"""
    
    def __init__(self, max_history: int = 20):
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.conversation_metadata[session_id] = {
                "created_at": time.time(),
                "message_count": 0,
                "last_updated": time.time()
            }
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        
        # Update metadata
        self.conversation_metadata[session_id]["message_count"] += 1
        self.conversation_metadata[session_id]["last_updated"] = time.time()
        
        # Trim history if too long (keep recent messages)
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history*2:]
    
    def get_history(self, session_id: str, include_metadata: bool = False) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        if session_id not in self.conversations:
            return []
        
        if include_metadata:
            return self.conversations.get(session_id, [])
        else:
            # Return only role and content for LLM context
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.conversations.get(session_id, [])
            ]
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get metadata about a conversation session"""
        return self.conversation_metadata.get(session_id, {})
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with metadata"""
        sessions = []
        for session_id, metadata in self.conversation_metadata.items():
            sessions.append({
                "session_id": session_id,
                "created_at": metadata["created_at"],
                "last_updated": metadata["last_updated"],
                "message_count": metadata["message_count"],
                "preview": self._get_session_preview(session_id)
            })
        # Sort by last updated (most recent first)
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)
        return sessions
    
    def _get_session_preview(self, session_id: str) -> str:
        """Get a preview of the conversation (first user message)"""
        messages = self.conversations.get(session_id, [])
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                return content[:50] + "..." if len(content) > 50 else content
        return "New conversation"
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.conversation_metadata:
            del self.conversation_metadata[session_id]
    
    def delete_old_sessions(self, days: int = 7):
        """Delete sessions older than specified days"""
        current_time = time.time()
        threshold = days * 24 * 60 * 60
        
        sessions_to_delete = []
        for session_id, metadata in self.conversation_metadata.items():
            if current_time - metadata["last_updated"] > threshold:
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            self.clear_session(session_id)
