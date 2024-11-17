import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
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
        self.vector_store = None  # Will be set after initialization
    
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
    
    def get_conversation_chain(self, chat_session: ChatSession):
        """Get or create a conversation chain for a chat session."""
        try:
            if chat_session.id in self._chain_cache:
                return self._chain_cache[chat_session.id]
            
            # Get vector store for this chat
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
                
            vectorstore = self.vector_store.get_retriever(chat_session.id)
            if not vectorstore:
                raise ValueError("Could not create retriever for chat session")
            
            # Create memory with updated configuration
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Custom prompt template for better source attribution
            prompt = PromptTemplate(
                template="""You are a helpful assistant for analyzing documents. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Please provide a detailed answer based on the context provided. Do not include any text about sources in your answer. The sources will be displayed separately below your answer.

Answer: """,
                input_variables=["context", "question", "chat_history"]
            )
            
            # Create QA chain with source documents
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,  # Important: Return source documents
                verbose=True
            )
            
            self._chain_cache[chat_session.id] = qa_chain
            return qa_chain
            
        except Exception as e:
            print(f"Error creating conversation chain: {str(e)}")
            raise ValueError(f"Failed to create conversation chain: {str(e)}")
    
    def add_message(self, chat: ChatSession, role: str, content: str):
        """Add a message to both display history and chat memory."""
        # Add to display messages with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # If it's an assistant response, split into answer and sources
        if role == "assistant" and "Sources:" in content:
            parts = content.split("Sources:", 1)
            answer = parts[0].replace("Answer:", "").strip()
            sources = "Sources:\n" + parts[1].strip() if len(parts) > 1 else ""
            
            chat.messages.append({
                "role": role,
                "content": answer,
                "sources": sources,
                "timestamp": timestamp
            })
        else:
            chat.messages.append({
                "role": role,
                "content": content,
                "timestamp": timestamp
            })
        
        # Add to chat history
        if role == "user":
            chat.chat_history.append(HumanMessage(content=content))
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