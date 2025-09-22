#!/usr/bin/env python3
"""
Simple test script for RAG system
"""

from rag_system import RAGSystem
import os

def test_rag_system():
    print("🚀 Testing RAG System...")
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=200
    )
    
    # Check if documents directory exists
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        print(f"❌ Documents directory '{docs_dir}' not found!")
        print("Please create the 'documents' folder and add some .txt files.")
        return
    
    # Check if we have any text files
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ No .txt files found in '{docs_dir}'!")
        print("Please add some .txt files to the documents folder.")
        return
    
    print(f"✅ Found {len(txt_files)} text files: {txt_files}")
    
    # Process documents or load existing data
    if os.path.exists("doc_store.json") and os.path.exists("vector_store.json"):
        print("📁 Loading existing processed data...")
        rag.load_data()
    else:
        print("🔄 Processing documents for the first time...")
        rag.process_documents_pipeline(docs_dir)
    
    # Test queries
    test_queries = [
        "What is this document about?",
        "Can you summarize the main points?",
        "What are the key topics discussed?"
    ]
    
    print("\n🎯 Testing queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        try:
            result = rag.query(query, top_k=2)
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"📊 Found {len(result['relevant_chunks'])} relevant chunks")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎉 RAG system test completed!")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        print("The embedding and retrieval parts will work, but LLM response generation will fail.")
        print()
    
    test_rag_system()