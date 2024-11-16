from typing import List, Optional
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
        )
    
    def create_index(self, index_name: str) -> None:
        """Create a new Pinecone index for a chat session."""
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
    
    def delete_index(self, index_name: str) -> None:
        """Delete a Pinecone index when chat is deleted."""
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
    
    def add_documents(self, documents: List[Document], index_name: str) -> None:
        """Add documents to the specified index."""
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings
        )
        vector_store.add_documents(documents)
    
    def get_retriever(self, index_name: str):
        """Get a retriever for the specified index."""
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings
        )
        return vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )