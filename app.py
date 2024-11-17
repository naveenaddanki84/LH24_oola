import os
import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.summarizer import DocumentSummarizer
from utils.chat_manager import ChatManager
from dotenv import load_dotenv
import time

# Load the environment variables
load_dotenv()

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStoreManager()
summarizer = DocumentSummarizer()
chat_manager = ChatManager()
chat_manager.vector_store = vector_store  # Set the vector store

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

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

def display_chat_messages():
    """Display chat messages with custom styling."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                st.markdown("---")
                st.markdown("📚 Top 3 Most Relevant Sources:")
                # Display each source in an expandable section
                for idx, source in enumerate(message["sources"], 1):
                    with st.expander(f"📄 Source {idx} - {source['file_name']} (Page: {source['page']})"):
                        st.markdown(source['text'])

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
    if not hasattr(current_chat, 'individual_summaries') or not current_chat.individual_summaries:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['txt', 'pdf', 'docx', 'csv', 'md', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'gif'],
            accept_multiple_files=True,
            help="Supported formats: Text files, PDFs, Word docs, Excel files, and images"
        )
        
        # Submit button for processing
        if uploaded_files:
            total_size = sum(file.size for file in uploaded_files)
            if total_size > 100 * 1024 * 1024:  # 100MB limit
                st.error("Total file size exceeds 100MB limit. Please reduce the size or number of files.")
                return
                
            st.write(f"Selected {len(uploaded_files)} files")
            if st.button("Process Documents"):
                with st.spinner('Processing documents...'):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process files and get summaries
                        temp_paths = []
                        filenames = []
                        
                        # Step 1: Save files
                        status_text.text("Saving uploaded files...")
                        for i, uploaded_file in enumerate(uploaded_files):
                            temp_path = doc_processor.save_temp_file(uploaded_file)
                            temp_paths.append(temp_path)
                            filenames.append(uploaded_file.name)
                            progress_bar.progress((i + 1) / (len(uploaded_files) * 4))
                        
                        # Step 2: Process documents
                        status_text.text("Processing documents...")
                        docs, raw_text = doc_processor.process_documents(temp_paths)
                        progress_bar.progress(0.5)
                        
                        # Step 3: Generate summaries
                        status_text.text("Generating summaries...")
                        summaries = summarizer.summarize_documents(raw_text, filenames)
                        progress_bar.progress(0.75)
                        
                        # Step 4: Store in vector store
                        status_text.text("Creating vector store...")
                        vector_store.create_index(current_chat.id)
                        vector_store.add_documents(docs, current_chat.id)
                        
                        # Update chat session
                        current_chat.individual_summaries = summaries['individual_summaries']
                        
                        # Cleanup temp files
                        status_text.text("Cleaning up...")
                        for temp_path in temp_paths:
                            doc_processor.cleanup_temp_file(temp_path)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")
                        st.success("Documents processed successfully!")
                        time.sleep(1)  # Give user time to see success message
                        st.rerun()
                        
                    except Exception as e:
                        # Cleanup on error
                        for temp_path in temp_paths:
                            doc_processor.cleanup_temp_file(temp_path)
                        st.error(f"Error processing documents: {str(e)}")
                        
                        # Log error for debugging
                        print(f"Document processing error: {str(e)}")
                        
    # Display summaries section
    if hasattr(current_chat, 'individual_summaries') and current_chat.individual_summaries:
        st.header("📑 Document Summaries")
        # Create tabs for each file
        tabs = st.tabs([f"📄 {summary['filename']}" for summary in current_chat.individual_summaries])
        for tab, summary in zip(tabs, current_chat.individual_summaries):
            with tab:
                st.write(summary['summary'])
    
    # Chat interface
    if hasattr(current_chat, 'individual_summaries') and current_chat.individual_summaries:
        st.header("💬 Chat")
        
        # Display messages
        if st.session_state.messages:
            display_chat_messages()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            with st.spinner('Thinking...'):
                try:
                    # Get conversation chain
                    chain = chat_manager.get_conversation_chain(current_chat)
                    
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Get response
                    response = chain({"question": prompt})
                    answer = response["answer"]
                    source_documents = response.get("source_documents", [])
                    
                    # Process source documents (limit to top 3)
                    sources = []
                    for idx, doc in enumerate(source_documents[:3], 1):  # Only process top 3 sources
                        sources.append({
                            'text': doc.page_content,
                            'file_name': os.path.basename(doc.metadata.get('source', 'Unknown')),
                            'page': doc.metadata.get('page', 'N/A'),
                            'score': doc.metadata.get('score', 0)  # Add score if available
                        })
                    
                    # Sort sources by score if available
                    sources.sort(key=lambda x: x.get('score', 0), reverse=True)
                    
                    # Add assistant message with sources
                    message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                    st.session_state.messages.append(message)
                    
                    # Rerun to update UI
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
                    print(f"Chat error: {str(e)}")  # Log error for debugging

if __name__ == "__main__":
    main()