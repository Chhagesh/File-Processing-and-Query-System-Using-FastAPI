# File-Processing-and-Query-System-Using-FastAPI

This FastAPI application implements a powerful RAG (Retrieval Augmented Generation) system for PDF document processing and intelligent querying. Key features include:

PDF processing with text extraction and document chunking
Vector storage using Chroma DB for efficient document retrieval
LangChain integration for context-aware question answering
Chat history management for contextual conversations
RESTful API endpoints for document upload and querying
CORS support for cross-origin requests
Secure file handling with temporary storage
HuggingFace embeddings integration
Structured JSON responses with timestamps and conversation IDs
The system allows users to upload multiple PDFs, process them into searchable vectors, and perform natural language queries with context-aware responses. Perfect for building document-based chatbots and Q&A systems.

Here's how this RAG (Retrieval Augmented Generation) system works:

Document Processing Flow:
Users upload PDF files through the /process endpoint
The system extracts text from each PDF page using PyPDF2
Documents are split and stored with metadata (filename, page numbers)
Text is converted to vector embeddings using HuggingFace
Vectors are stored in a Chroma database for efficient retrieval
Query Processing Flow:
Users send questions through the /query endpoint
The system uses a history-aware retriever to:
Consider chat history for context
Reformulate questions when needed
Retrieve relevant document chunks
The RAG chain combines:
Retrieved context
User question
Chat history
Generates concise, contextual answers (max 3 sentences)
Technical Components:
FastAPI handles HTTP requests and file uploads
LangChain manages the RAG pipeline
Chroma DB provides vector storage and similarity search
Chat history tracks conversation flow
Responses include timestamps and conversation IDs
The system is designed for real-time document-based Q&A with context awareness, making it ideal for interactive document exploration and information retrieval.
