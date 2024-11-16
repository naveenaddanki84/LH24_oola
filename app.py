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
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for styling chat messages
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
    .disabled-button {
        pointer-events: none;
        opacity: 0.5;
        cursor: not-allowed;
    }
</style>
""", unsafe_allow_html=True)

def handle_file_upload(files, chat_id: str) -> str:
    """Process uploaded files and return a summary."""
    temp_paths = []
    try:
        for file in files:
            temp_path = doc_processor.save_temp_file(file)
            temp_paths.append(temp_path)
        
        docs, raw_text = doc_processor.process_documents(temp_paths)
        summary = summarizer.summarize_documents(raw_text)
        vector_store.create_index(chat_id)
        vector_store.add_documents(docs, chat_id)
        
        return summary
    
    finally:
        for temp_path in temp_paths:
            doc_processor.cleanup_temp_file(temp_path)

def display_chat_messages():
    """Display chat messages with custom styling."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.title("📚 Document Chat Assistant")
    
    # Sidebar for chat management
    with st.sidebar:
        st.header("Chat Sessions")
        
        if st.button("New Chat Session", key="new_chat"):
            chat = chat_manager.create_chat()
            st.session_state.chats[chat.id] = chat
            st.session_state.current_chat_id = chat.id
            st.session_state.uploaded_files = []
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        for chat_id, chat in st.session_state.chats.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"💬 {chat.created_at}", key=f"select_{chat_id}") and not st.session_state.uploaded_files:
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    chat_manager.clear_chat_history(chat)
                    vector_store.delete_index(chat_id)
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = None
                    st.rerun()

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
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files is not None:
            st.session_state.uploaded_files = list(uploaded_files)
        else:
            st.session_state.uploaded_files = []

        submit_button_disabled = not st.session_state.uploaded_files
        submit_button = st.button("Submit", disabled=submit_button_disabled)

        if submit_button and st.session_state.uploaded_files:
            with st.spinner('Processing documents...'):
                try:
                    summary = handle_file_upload(st.session_state.uploaded_files, current_chat.id)
                    chat_manager.set_summary(current_chat, summary)
                    st.session_state.uploaded_files = []
                    st.success("Documents processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Display the summary if available and keep it expanded by default
    if current_chat.summary:
        with st.expander("📑 Document Summary", expanded=True):  # Automatically expanded
            st.markdown(current_chat.summary)
        
        st.header("💬 Chat")
        display_chat_messages()
        
        # Show chat input only after the summary is generated
        prompt = st.chat_input("Ask a question about your documents...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            retriever = vector_store.get_retriever(current_chat.id)
            chain = chat_manager.get_conversation_chain(retriever, current_chat)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chain.invoke({"question": prompt})
                    assistant_reply = response['answer']
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                    st.markdown(assistant_reply)
    else:
        st.info("📄 Please upload and process your documents to start asking questions.")

if __name__ == "__main__":
    main()
