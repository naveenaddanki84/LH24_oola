from typing import List
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

class DocumentSummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        
        # Enhanced prompt for detailed summaries
        self.map_prompt_template = """Generate a comprehensive summary of the following text. Include:
1. Main Topic/Purpose:
   - Central theme or objective
   - Key context or background

2. Key Points:
   - Major arguments or findings
   - Important data or statistics
   - Significant conclusions

3. Supporting Details:
   - Relevant examples
   - Evidence or justifications
   - Important relationships or connections

4. Additional Information:
   - Notable quotes or references
   - Technical terms or definitions
   - Special considerations

Text to summarize:
"{text}"

Detailed Summary:"""
        
        self.combine_prompt_template = """Create a comprehensive summary of the following document summaries:
"{text}"

Please structure the summary as follows:
1. Overview
   - Main themes and objectives
   - Document scope and context

2. Key Findings
   - Major points from all documents
   - Common themes or patterns
   - Important differences or contrasts

3. Critical Details
   - Supporting evidence
   - Important examples
   - Relevant data points

4. Conclusions
   - Main takeaways
   - Implications or recommendations
   - Areas for further consideration

Comprehensive Summary:"""
    
    def summarize_single_document(self, text: str, filename: str) -> str:
        """Generate summary for a single document."""
        try:
            doc = Document(page_content=text)
            map_prompt = PromptTemplate(
                template=self.map_prompt_template,
                input_variables=["text"]
            )
            
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=map_prompt
            )
            
            summary = chain.invoke({"input_documents": [doc]})
            return {"filename": filename, "summary": summary["output_text"].strip()}
        except Exception as e:
            print(f"Error summarizing {filename}: {str(e)}")
            return {"filename": filename, "summary": f"Error generating summary: {str(e)}"}
    
    def summarize_documents(self, raw_text: List[str], filenames: List[str] = None) -> dict:
        """Generate summaries for all documents."""
        try:
            if not filenames:
                filenames = [f"Document_{i+1}" for i in range(len(raw_text))]
            
            # Generate individual summaries
            individual_summaries = []
            for text, filename in zip(raw_text, filenames):
                summary = self.summarize_single_document(text, filename)
                individual_summaries.append(summary)
            
            return {
                "individual_summaries": individual_summaries
            }
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return {
                "individual_summaries": []
            }

