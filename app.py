import os
import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.summarizer import DocumentSummarizer
from utils.chat_manager import ChatManager
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStoreManager()
summarizer = DocumentSummarizer()
chat_manager = ChatManager()

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None

# Page config
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .assistant-message {
        background-color: #f0f0f0;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def handle_file_upload(files, chat_id: str) -> str:
    """Process uploaded files and return summary."""
    temp_paths = []
    try:
        # Save files temporarily
        for file in files:
            temp_path = doc_processor.save_temp_file(file)
            temp_paths.append(temp_path)
        
        # Process documents
        docs, raw_text = doc_processor.process_documents(temp_paths)
        
        # Generate summary
        summary = summarizer.summarize_documents(raw_text)
        
        # Store embeddings
        vector_store.create_index(chat_id)
        vector_store.add_documents(docs, chat_id)
        
        return summary
    
    finally:
        # Cleanup temporary files
        for temp_path in temp_paths:
            doc_processor.cleanup_temp_file(temp_path)

def display_chat_messages(chat):
    """Display chat messages with custom styling."""
    for message in chat.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.title("📚 Document Chat Assistant")
    
    # Sidebar for chat management
    with st.sidebar:
        st.header("Chat Sessions")
        
        # New chat button
        if st.button("New Chat Session", key="new_chat"):
            chat = chat_manager.create_chat()
            st.session_state.chats[chat.id] = chat
            st.session_state.current_chat_id = chat.id
            st.rerun()
        
        st.divider()
        
        # List all chats
        for chat_id, chat in st.session_state.chats.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"💬 {chat.created_at}", key=f"select_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    # Clear chat history before deleting
                    chat_manager.clear_chat_history(chat)
                    vector_store.delete_index(chat_id)
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = None
                    st.rerun()
    
    # Main chat interface
    if st.session_state.current_chat_id is None:
        st.info("👈 Please create a new chat session from the sidebar to begin.")
        return
    
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Document upload section (only show if no documents processed yet)
    if not current_chat.summary:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['txt', 'pdf', 'docx', 'csv', 'md', 'xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner('Processing documents...'):
                try:
                    summary = handle_file_upload(uploaded_files, current_chat.id)
                    chat_manager.set_summary(current_chat, summary)
                    st.success("Documents processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    else:
        # Show document summary
        with st.expander("📑 Document Summary", expanded=False):
            st.markdown(current_chat.summary)
        
        # Chat interface
        st.header("💬 Chat")
        display_chat_messages(current_chat)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            chat_manager.add_message(current_chat, "user", prompt)
            
            # Get response from RAG
            retriever = vector_store.get_retriever(current_chat.id)
            chain = chat_manager.get_conversation_chain(retriever, current_chat)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chain.invoke({"question": prompt})
                    chat_manager.add_message(
                        current_chat, "assistant", response['answer']
                    )
                    st.markdown(response['answer'])

if __name__ == "__main__":
    main()