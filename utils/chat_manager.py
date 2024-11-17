import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from functools import lru_cache
from langchain.chains.question_answering import load_qa_chain
import re

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
        # Define the custom system prompt
        self.system_prompt = """You are a helpful assistant specialized in answering questions based on provided documents.

Sensitive information includes but is not limited to passwords, phone numbers, email addresses, API keys, SSNs, or any information that could be harmful to individuals.

If a user asks for sensitive information, always respond with:
"Sorry, I cannot provide you the details you asked as it contains sensitive information."

Aggressive behavior includes rude language, insults, or blaming.

When a user uses aggressive language, respond gently by acknowledging their feelings and providing helpful information if possible.

Always provide clear and concise answers based on the documents.

If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

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

            # Custom prompt template with system prompt
            prompt = PromptTemplate(
                template=f"""{self.system_prompt}

Context from documents:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer: """,
                input_variables=["context", "question", "chat_history"]
            )

            # Create the chain to combine documents
            doc_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt,
                document_variable_name="context"
            )

            # Create a prompt for the question generator
            question_generator_prompt = PromptTemplate.from_template(
                "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow-up Question: {question}\nStandalone Question:"
            )

            # Create the question generator chain
            question_generator = LLMChain(
                llm=self.llm,
                prompt=question_generator_prompt
            )

            # Create the Conversational Retrieval Chain
            chain = ConversationalRetrievalChain(
                retriever=vectorstore,
                memory=memory,
                combine_docs_chain=doc_chain,
                question_generator=question_generator,
                verbose=False,
                return_source_documents=True
            )

            self._chain_cache[chat_session.id] = chain
            return chain

        except Exception as e:
            print(f"Error creating conversation chain: {str(e)}")
            raise ValueError(f"Failed to create conversation chain: {str(e)}")

    def add_message(self, chat: ChatSession, role: str, content: str):
        """Add a message to both display history and chat memory."""
        # Add to display messages with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
 
