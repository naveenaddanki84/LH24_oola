import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

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
        # Define the custom prompt
        self.system_prompt = """You are a helpful assistant specialized in answering questions based on provided documents.

Sensitive information includes but is not limited to passwords, phone numbers, email addresses, API keys, or any information that could be harmful to individuals.

If a user asks for sensitive information, always respond with:
"Sorry, I cannot provide you the details you asked as it contains sensitive information."

Aggressive behavior includes rude language, insults, or blaming.

When a user uses aggressive language, respond gently by acknowledging their feelings and provide helpful information if possible.

Always provide clear and concise answers based on the documents.
"""

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
        """Create a conversation chain for RAG with persistent memory and custom prompt."""
        message_history = CustomChatMessageHistory(chat_session)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            output_key="answer"
        )
        
        # Create the prompt templates
        qa_prompt_template = f"""{self.system_prompt}

{{context}}

Question: {{question}}

Answer:"""

        # Ensure "context" and "question" are input variables
        qa_prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )

        # Create the chain to combine documents
        doc_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=qa_prompt,
            document_variable_name="context"  # Ensure this matches "context" in the prompt
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
            retriever=retriever,
            memory=memory,
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            verbose=False,
            return_source_documents=True
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

    def detect_sensitive_question(self, question: str) -> bool:
        """Detect if the question contains sensitive information."""
        sensitive_keywords = [
            "password", "phone number", "email address", "api key", "credit card", "ssn", "secret", "contact", "contact information"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in sensitive_keywords)
