import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from functools import lru_cache

@dataclass
class ChatSession:
    id: str
    created_at: str
    messages: List[Dict[str, str]]
    summary: Optional[str] = None
    chat_history: List[BaseMessage] = field(default_factory=list)
    personal_context: Dict[str, Any] = field(default_factory=dict)  # Store personal info like name

class CustomChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, chat_session: ChatSession):
        self.chat_session = chat_session

    def add_message(self, message: BaseMessage) -> None:
        self.chat_session.chat_history.append(message)

    def clear(self) -> None:
        self.chat_session.chat_history = []

    @property
    def messages(self) -> List[BaseMessage]:
        return self.chat_session.chat_history

class ChatManager:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self._chain_cache = {}  # Cache for conversation chains
    
    @lru_cache(maxsize=10)
    def _get_cached_chain(self, chat_id: str):
        """Get cached conversation chain."""
        return self._chain_cache.get(chat_id)
    
    def create_chat(self) -> ChatSession:
        """Create a new chat session."""
        return ChatSession(
            id=str(uuid.uuid4()),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            messages=[],
            summary=None,
            chat_history=[],
            personal_context={}
        )
    
    def get_conversation_chain(self, retriever, chat_session: ChatSession):
        """Create a conversation chain for RAG with persistent memory."""
        # Check cache first
        cached_chain = self._get_cached_chain(chat_session.id)
        if cached_chain:
            return cached_chain
            
        message_history = CustomChatMessageHistory(chat_session)
        
        # Window memory for recent conversations
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            output_key="answer",
            k=20  # Increased for better context retention
        )
        
        # Create a comprehensive system message
        system_message = """You are a helpful assistant deals with QA from the documents uploaded.

Key points to remember:
1. Stay focused on the context
2. Maintain conversation context and refer back to previous discussions
3. If the user goes off-topic, gently guide them back to the project focus
4. Be consistent with your responses and maintain knowledge of previous interactions"""

        if chat_session.personal_context:
            system_message += "\n\nUser Context:"
            for key, value in chat_session.personal_context.items():
                system_message += f"\n- {key}: {value}"

        if chat_session.summary:
            system_message += "\n\nDocument Context:\n" + chat_session.summary
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            return_source_documents=True,
            rephrase_question=False,
            combine_docs_chain_kwargs={
                "prompt": None,
                "document_variable_name": "context"
            }
        )

        # Add system message to memory
        memory.chat_memory.add_message(SystemMessage(content=system_message))
        
        # Cache the chain
        self._chain_cache[chat_session.id] = chain
        return chain
    
    def add_message(self, chat: ChatSession, role: str, content: str):
        """Add a message to both display history and chat memory."""
        # Add to display messages with timestamp for better context tracking
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat.messages.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        
        if role == "user":
            # Update personal context
            if "my name is" in content.lower():
                name = content.lower().split("my name is")[-1].strip()
                chat.personal_context["name"] = name
            elif "i am" in content.lower():
                activity = content.lower().split("i am")[-1].strip()
                chat.personal_context["current_activity"] = activity
            
            # Create message with context if needed
            message = content
            if chat.summary and not any(kw in content.lower() for kw in chat.summary.lower().split()):
                message += "\n\nNote: Let me help you with questions about the uploaded documents."
            
            chat.chat_history.append(HumanMessage(content=message))
        else:
            chat.chat_history.append(AIMessage(content=content))
        
        # Refresh chain if significant context changes occurred
        if role == "user" and (
            len(chat.personal_context) > 0 or 
            "document" in content.lower() or 
            "context" in content.lower()
        ):
            if chat.id in self._chain_cache:
                del self._chain_cache[chat.id]
    
    def set_summary(self, chat: ChatSession, summary: str):
        """Set the document summary for the chat session."""
        chat.summary = summary
    
    def clear_chat_history(self, chat: ChatSession):
        """Clear chat history when deleting a chat."""
        chat.messages.clear()
        chat.chat_history.clear()