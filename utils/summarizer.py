from typing import List
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

class DocumentSummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        
        # Custom prompt for better summaries
        self.map_prompt_template = """Write a concise summary of the following text:
        "{text}"
        CONCISE SUMMARY:"""
        
        self.combine_prompt_template = """Write a comprehensive summary of the following document summaries:
        "{text}"
        
        Focus on the main points and ensure the summary is well-organized.
        
        COMPREHENSIVE SUMMARY:"""
    
    def summarize_documents(self, raw_text: List[str]) -> str:
        """Generate a combined summary of all documents."""
        try:
            # Create documents from raw text
            docs = [Document(page_content=text) for text in raw_text]
            
            # Create map and combine prompts
            map_prompt = PromptTemplate(
                template=self.map_prompt_template,
                input_variables=["text"]
            )
            
            combine_prompt = PromptTemplate(
                template=self.combine_prompt_template,
                input_variables=["text"]
            )
            
            # Create and run the summarization chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=False
            )
            
            # Generate summary using the new invoke method
            result = chain.invoke(docs)
            return result['output_text'].strip()
            
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")