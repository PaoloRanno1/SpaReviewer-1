# SPA Analysis Tool

## Overview

This project is a Streamlit-based web application designed for analyzing SPA (Stock Purchase Agreement) documents using advanced AI capabilities. The system combines document processing, semantic chunking, and Google's Generative AI to provide comprehensive analysis of legal documents. The application leverages vector databases for efficient document storage and retrieval, enabling sophisticated document analysis workflows.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **UI Components**: Sidebar navigation, file upload interface, and results display
- **Page Configuration**: Responsive design with expanded sidebar state and custom page icon

### Document Processing Pipeline
- **PDF Loading**: PyMuPDFLoader for robust PDF document extraction
- **Text Chunking**: Dual-strategy approach combining semantic and recursive chunking
  - Semantic chunking using breakpoint threshold analysis (percentile-based at 0.8 threshold)
  - Fallback recursive character text splitter (2000 char chunks, 200 char overlap)
  - Minimum chunk size enforcement (1100 characters) with intelligent merging
- **Error Handling**: Custom exception classes (ChunkingError, EmbeddingError) for granular error management

### AI and Embeddings
- **Embedding Model**: Google Generative AI Embeddings (GoogleGenerativeAIEmbeddings)
- **Default Model**: Gemini-2.5-flash for document analysis
- **Vector Storage**: Chroma database for efficient similarity search and retrieval

### Configuration Management
- **Multi-tier Configuration**: Environment variables with fallback hierarchy
  1. .env file (python-dotenv integration)
  2. Streamlit secrets management
  3. System environment variables
- **Key Parameters**: Google API key, database directory, model selection

### Data Storage
- **Vector Database**: Chroma for document embeddings and metadata storage
- **File System**: Local directory-based storage (DATABASE_SPA default)
- **Document Metadata**: SPA name tracking and document provenance

## External Dependencies

### Core AI Services
- **Google Generative AI**: Primary embedding and analysis service requiring API key authentication
- **LangChain Framework**: Document processing, text splitting, and vector database integration
  - langchain_community: Document loaders and utilities
  - langchain_experimental: Advanced text processing features
  - langchain_google_genai: Google AI service integration
  - langchain_chroma: Vector database operations

### Document Processing
- **PyMuPDF**: PDF document parsing and text extraction
- **python-dotenv**: Environment variable management (optional dependency)

### Web Framework
- **Streamlit**: Complete web application framework for UI and deployment

### Python Standard Libraries
- **logging**: Application logging and debugging
- **pathlib**: Modern file system path handling
- **json**: Configuration and data serialization
- **os**: System environment access
- **typing**: Type hints for code clarity and IDE support