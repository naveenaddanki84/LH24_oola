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
import base64
from openai import OpenAI

class ImageProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image(self, image_path: str) -> str:
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please describe this image in detail, including any visible text, objects, or important elements."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            if not response.choices:
                return "Error: No response from image analysis"
                
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"Error in image analysis: {error_msg}")
            if "model_not_found" in error_msg:
                return ("Error: Could not access the image analysis model. Please check:\n"
                       "1. Your OpenAI API key is valid\n"
                       "2. The model name is correct")
            return f"Error analyzing image: {error_msg}"

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        self.image_processor = ImageProcessor()
    
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
        
        # Handle image files
        if extension.lower() in ['jpg', 'jpeg', 'png', 'gif']:
            return None  # We'll handle images differently
        
        if extension not in loaders:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return loaders[extension](file_path)
    
    def process_documents(self, files: List[str]) -> Tuple[List[Document], List[str]]:
        """Process multiple documents and return chunks and raw text."""
        all_docs = []
        raw_text = []
        errors = []
        
        for file_path in files:
            try:
                extension = file_path.split('.')[-1].lower()
                
                # Handle image files
                if extension in ['jpg', 'jpeg', 'png', 'gif']:
                    image_description = self.image_processor.analyze_image(file_path)
                    if image_description.startswith("Error"):
                        errors.append(f"{os.path.basename(file_path)}: {image_description}")
                        continue
                        
                    doc = Document(
                        page_content=image_description,
                        metadata={"source": file_path, "type": "image"}
                    )
                    all_docs.append(doc)
                    raw_text.append(f"Image Analysis ({os.path.basename(file_path)}):\n{image_description}")
                    continue
                
                # Handle other document types
                loader = self.get_loader(file_path)
                if loader is None:
                    errors.append(f"Unsupported file format: {extension}")
                    continue
                    
                docs = loader.load()
                
                # Store raw text for summarization
                raw_text.extend([doc.page_content for doc in docs])
                
                # Split documents into chunks
                if docs:
                    split_docs = self.text_splitter.split_documents(docs)
                    all_docs.extend(split_docs)
            
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing {file_path}: {error_msg}")
                errors.append(f"{os.path.basename(file_path)}: {error_msg}")
                continue
        
        if errors:
            error_summary = "\n".join(errors)
            print(f"Processing completed with errors:\n{error_summary}")
        
        return all_docs, raw_text
    
    def save_temp_file(self, uploaded_file) -> str:
        """Save uploaded file temporarily and return path."""
        # Create temp directory if it doesn't exist
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Create unique filename to avoid conflicts
        temp_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    
    def cleanup_temp_file(self, temp_path: str):
        """Remove temporary file."""
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                # Remove temp directory if empty
                temp_dir = os.path.dirname(temp_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_path}: {str(e)}")

