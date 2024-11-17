import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

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
        message_history = CustomChatMessageHistory(chat_session)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            output_key="answer"
        )

#         # Define a custom prompt template without source citation instructions
#         prompt_template = """
# You are an assistant that helps answer questions based on the following documents.

# {context}

# Question: {question}

# Answer:
# """
#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=["context", "question"]
#         )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            verbose=False,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": None}
        )

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
