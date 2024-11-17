import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from functools import lru_cache

@dataclass
class ChatSession:
    id: str
    created_at: str
    messages: List[Dict[str, str]]
    summary: Optional[str] = None
    chat_history: List[BaseMessage] = field(default_factory=list)

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
            chat_history=[]
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
            output_key="answer",  # Explicitly set output key
            k=10  # Increased window size to maintain more context
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            verbose=True,  # Enable verbose mode for debugging
            return_source_documents=True,
            rephrase_question=False,  # Disable automatic question rephrasing
            combine_docs_chain_kwargs={
                "prompt": None,
                "document_variable_name": "context"
            }
        )
        
        # Cache the chain
        self._chain_cache[chat_session.id] = chain
        return chain
    
    def add_message(self, chat: ChatSession, role: str, content: str):
        """Add a message to both display history and chat memory."""
        # Add to display messages
        chat.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Add to chat history
        if role == "user":
            chat.chat_history.append(HumanMessage(content=content))
        else:
            chat.chat_history.append(AIMessage(content=content))
    
    def set_summary(self, chat: ChatSession, summary: str):
        """Set the document summary for the chat session."""
        chat.summary = summary
    
    def clear_chat_history(self, chat: ChatSession):
        """Clear chat history when deleting a chat."""
        chat.messages.clear()
        chat.chat_history.clear()