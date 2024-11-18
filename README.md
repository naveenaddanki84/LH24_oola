#  Multimodal Document Analysis and Q&A System

## 🎯 Project Overview
An advanced document processing and analysis platform designed for comprehensive document interaction and intelligent Q&A capabilities.

### 🌟 Key Features

#### 📊 Document Processing
- **Multi-Format Support**:
  - Text: PDF, DOCX, TXT, CSV, MD, XLSX, XLS
  - Images: JPG, JPEG, PNG, GIF
- **Intelligent Processing**: Advanced text extraction and processing
- **Temporary File Management**: Secure handling of uploaded documents

#### 📝 Enhanced Document Summarization
Our platform provides detailed, structured summaries at multiple levels:

**Individual Document Analysis**:
1. Main Topic/Purpose
   - Central theme identification
   - Key context and background
2. Key Points
   - Major arguments and findings
   - Important statistics
   - Significant conclusions
3. Supporting Details
   - Relevant examples
   - Evidence and justifications
   - Important relationships
4. Additional Information
   - Notable quotes
   - Technical terms
   - Special considerations

**Multi-Document Analysis**:
1. Overview
   - Main themes and objectives
   - Document scope and context
2. Key Findings
   - Major points across documents
   - Common patterns
   - Important contrasts
3. Critical Details
   - Supporting evidence
   - Important examples
   - Key data points
4. Conclusions
   - Main takeaways
   - Recommendations
   - Future considerations

#### 🔍 Vector Store Management
- Pinecone-based vector database
- OpenAI embeddings integration
- Semantic search capabilities
- Per-session indexing

#### 💬 Interactive Chat Interface
- Context-aware Q&A
- Source-matched responses
- Multiple chat session support
- Expandable source attribution

## Architecture Diagram
![image](https://github.com/user-attachments/assets/0a74e06e-ac2c-46f0-8309-a5f02e52a786)


## 🛠️ Technical Requirements

### Environment Variables
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

### Dependencies
```
langchain>=0.1.0
langchain-openai>=0.0.2
pinecone-client>=3.0.0
streamlit>=1.29.0
openai>=1.10.0
```

## 🚀 Getting Started

1. Clone the repository
2. Set up environment variables
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🔒 Security Features
- Sensitive question detection
- Temporary file cleanup
- API key protection
- Safe message handling

## 💡 Future Enhancements
- Enhanced file format support
- Advanced embedding techniques
- Caching mechanisms
- Improved retrieval methods

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
