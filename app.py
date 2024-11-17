import os
import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.summarizer import DocumentSummarizer
from utils.chat_manager import ChatManager
from dotenv import load_dotenv
import openai
import logging

# Load the environment variables
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

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
if 'file_summaries' not in st.session_state:
    st.session_state.file_summaries = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO)

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
    .disabled-button {
        pointer-events: none;
        opacity: 0.5;
        cursor: not-allowed;
    }
</style>
""", unsafe_allow_html=True)

def handle_file_upload(files, chat_id: str) -> dict:
    """Process each uploaded file and return a dictionary of summaries."""
    summaries = {}
    for file in files:
        temp_paths = []
        try:
            # Save file temporarily
            temp_path = doc_processor.save_temp_file(file)
            temp_paths.append(temp_path)
            
            # Process document and generate summary
            docs, raw_text = doc_processor.process_documents([temp_path])
            summary = summarizer.summarize_documents(raw_text)
            
            # Store embeddings for each document
            vector_store.create_index(chat_id)
            vector_store.add_documents(docs, chat_id)
            
            # Store summary with filename as key
            summaries[file.name] = summary
        
        finally:
            # Cleanup temporary files
            for temp_path in temp_paths:
                doc_processor.cleanup_temp_file(temp_path)
    
    return summaries

def display_chat_messages():
    """Display chat messages with custom styling."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                # Display each source in its own expander
                for source in message["sources"]:
                    idx = source['source_number']
                    with st.expander(f"📄 Source {idx} (File: {source['file_path']}, Page: {source['page_number']})", expanded=False):
                        st.markdown(f"{source['text']}\n")

def is_thank_you_message(message: str) -> bool:
    """Check if the message is a thank-you message."""
    thank_you_phrases = [
        "thank you",
        "thanks",
        "thank you very much",
        "thanks a lot",
        "thank you so much",
        "thank u",
        "thx",
        "ty",
        "cheers",
        "much appreciated",
        "i appreciate it",
        "gracias",
        "danke",
        "merci",
        "thanks for your help"
    ]
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in thank_you_phrases)

def is_aggressive_message(message: str) -> bool:
    """Check if the message contains aggressive or negative language."""
    aggressive_phrases = [
        "you're useless",
        "this is stupid",
        "can't you do anything",
        "you are dumb",
        "idiot",
        "not helpful",
        "waste of time",
        "why can't you answer",
        "you are terrible",
        "you're awful",
        "pathetic"
    ]
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in aggressive_phrases)

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
            st.session_state.uploaded_files = []
            st.session_state.file_summaries = {}
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # List all chats
        for chat_id, chat in st.session_state.chats.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"💬 {chat.created_at}", key=f"select_{chat_id}") and not st.session_state.uploaded_files:
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = []
                    st.session_state.file_summaries = {}
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
    if not st.session_state.file_summaries:
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['txt', 'pdf', 'docx', 'csv', 'md', 'xlsx', 'xls'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = list(uploaded_files)
        else:
            st.session_state.uploaded_files = []
        
        submit_button_disabled = not st.session_state.uploaded_files
        submit_button = st.button("Submit", disabled=submit_button_disabled)
        
        if submit_button and st.session_state.uploaded_files:
            with st.spinner('Processing documents...'):
                try:
                    summaries = handle_file_upload(st.session_state.uploaded_files, current_chat.id)
                    st.session_state.file_summaries = summaries
                    st.session_state.uploaded_files = []
                    st.success("Documents processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    else:
        # Display summaries with separate expanders for each file
        st.header("📑 Document Summaries")
        for filename, summary in st.session_state.file_summaries.items():
            with st.expander(f"📄 {filename} Summary", expanded=True):
                st.markdown(summary)
        
        # Chat interface
        st.header("💬 Q&A")
        display_chat_messages()
        
        # Chat input
        prompt = st.chat_input("Ask a question about your documents...")
        if prompt is not None:
            # Validate input
            if not prompt.strip():
                st.error("Input cannot be empty. Please provide a valid question.")
            else:
                # Log user input
                logging.info(f"User Input: {prompt}")
                
                # Check if the message is a thank-you message
                if is_thank_you_message(prompt):
                    response_text = (
                        "You're very welcome! I'm glad I could assist you. "
                        "If you have any more questions or need further assistance, feel free to ask!"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                # Check for aggressive messages
                elif is_aggressive_message(prompt):
                    response_text = (
                        "I understand your frustration, and I’m sorry I couldn’t provide the information you were seeking. "
                        "However, due to the instructions I follow, I cannot share sensitive information.\n\n"
                        "Is there anything else I can help you with?"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                # Check for sensitive questions
                elif chat_manager.detect_sensitive_question(prompt):
                    response_text = (
                        "Sorry, I cannot provide you the details you asked as it contains sensitive information."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                else:
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Get response from RAG
                    retriever = vector_store.get_retriever(current_chat.id)
                    chain = chat_manager.get_conversation_chain(retriever, current_chat)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                response = chain({"question": prompt})
                                assistant_reply = response.get('answer', '').strip()
                                source_documents = response.get('source_documents', [])
                                
                                # Check if assistant reply is empty
                                if not assistant_reply:
                                    st.warning("No relevant documents found for your query.")
                                    assistant_reply = (
                                        "I'm sorry, I could not find an answer to your question in the documents."
                                    )
                                
                                # Append follow-up question
                                assistant_reply += "\n\nIs there anything else I can help you with?"
                                
                                # Process source documents to get texts and metadata
                                source_texts = []
                                for idx, doc in enumerate(source_documents, 1):
                                    text = doc.page_content
                                    metadata = doc.metadata
                                    file_path = metadata.get('file_path', 'Unknown file')
                                    page_number = metadata.get('page', 'Unknown page')
                                    source_texts.append({
                                        'text': text,
                                        'file_path': file_path,
                                        'page_number': page_number,
                                        'source_number': idx  # Add source number
                                    })
                                
                                # Store the assistant's reply and sources
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": assistant_reply,
                                    "sources": source_texts
                                })
                                
                                # Display the assistant's reply
                                st.markdown(assistant_reply)
                                
                                # Display each source in its own expander
                                for source in source_texts:
                                    idx = source['source_number']
                                    with st.expander(f"📄 Source {idx} (File: {source['file_path']}, Page: {source['page_number']})", expanded=False):
                                        st.markdown(f"{source['text']}\n")
                                
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
                                logging.error(f"Chain invocation error: {e}")
                    
if __name__ == "__main__":
    main()
