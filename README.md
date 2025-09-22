# RAG System - From Scratch
A simple Retrieval Augmented Generation (RAG) system built without LangChain or LlamaIndex to understand the core concepts.
What it does

Reads documents from a folder
Splits them into chunks for processing
Creates embeddings (vectors) for semantic search
Finds relevant content for your questions
Generates answers using OpenAI GPT

## Quick Setup
1. Install
bashpip install transformers torch numpy sentence-transformers openai python-dotenv
2. Add your OpenAI API key
Create .env file:
OPENAI_API_KEY=your_key_here
3. Add documents
Put your .txt files in the documents/ folder.
4. Run
bashpython test_rag.py
How to use
pythonfrom rag_system import RAGSystem

## Initialize
rag = RAGSystem()

## Process your documents (run once)
rag.process_documents_pipeline("documents")

## Ask questions
result = rag.query("What is machine learning?")
print(result['response'])

## Files
rag_system.py - Main RAG implementation
test_rag.py - Test the complete system
requirements.txt - Dependencies
documents/ - Put your text files here

## Models Used

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (converts text to vectors)
LLM: gpt-3.5-turbo (generates final answers)

## How it works
Your Documents → Text Chunks → Embeddings → Vector Search → GPT Response

Documents get split into smaller chunks
Each chunk becomes a 384-dimensional vector
Your question also becomes a vector
System finds most similar chunks using cosine similarity
Relevant chunks + your question go to GPT for the final answer

Example
Question: "What causes children to have tantrums?"
Retrieved chunks: Text about child behavior, emotions, etc.
GPT Response: "Based on the retrieved information, children throw tantrums when they are overwhelmed by strong emotions, overstimulated, or anxious..."
Troubleshooting

OpenAI quota exceeded: Add credits to your OpenAI account
No documents found: Put .txt files in documents/ folder
Import errors: Make sure virtual environment is activated

License
MIT License - feel free to learn and modify!