import os
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
    
    def get_loader(self, file_path: str):
        """Get appropriate loader based on file extension."""
        extension = file_path.split('.')[-1].lower()
        loaders = {
            "txt": TextLoader,
            "pdf": PDFPlumberLoader,
            "docx": UnstructuredWordDocumentLoader,
            "csv": CSVLoader,
            "md": UnstructuredMarkdownLoader,
            "xlsx": UnstructuredExcelLoader,
            "xls": UnstructuredExcelLoader
        }
        
        if extension not in loaders:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return loaders[extension](file_path)
    
    def process_documents(self, files: List[str]) -> Tuple[List[Document], List[str]]:
        """Process multiple documents and return chunks and raw text."""
        all_docs = []
        raw_text = []
        
        for file_path in files:
            try:
                # Load document
                loader = self.get_loader(file_path)
                docs = loader.load()
                
                # Store raw text for summarization
                raw_text.extend([doc.page_content for doc in docs])
                
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(docs)
                all_docs.extend(split_docs)
                
            except Exception as e:
                raise Exception(f"Error processing {file_path}: {str(e)}")
        
        return all_docs, raw_text
    
    def save_temp_file(self, uploaded_file) -> str:
        """Save uploaded file temporarily and return path."""
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    
    def cleanup_temp_file(self, temp_path: str):
        """Remove temporary file."""
        if os.path.exists(temp_path):
            os.remove(temp_path)