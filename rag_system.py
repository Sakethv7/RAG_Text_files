import os
import re
import uuid # For unique IDs
import json
import numpy as np
import torch # For tensor operations
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer # sentence transformers for embeddings are used for better performance to embed text into vectors
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv() #  LLM API keys should be set here from the environment file

class RAGSystem:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=200): # Updated default embedding model
        """
        Initialize RAG system with updated model names
        
        Updated embedding models to choose from:
        - "sentence-transformers/all-MiniLM-L6-v2" (lightweight, fast)
        - "sentence-transformers/all-mpnet-base-v2" (better quality)
        - "BAAI/bge-small-en-v1.5" (good performance)
        - "BAAI/bge-base-en-v1.5" (better quality)
        - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
        """
        self.embedding_model_name = embedding_model_name # Store model name
        self.chunk_size = chunk_size # Max tokens per chunk
        
        # Initialize embedding model (using sentence-transformers for better performance)
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name) # Using SentenceTransformer for embeddings
        
        # For tokenization (if needed for chunking)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Using BERT tokenizer for chunking
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY")) # Ensure your OpenAI key is set in environment variables
        
        # Storage
        self.documents = {} # doc_id -> {text, metadata}
        self.vector_store = {} # chunk_id -> embedding vector
    
    def read_documents(self, directory_path): # this function reads text files from a directory
        
        """Read documents from directory"""
        
        print(f"Reading documents from: {directory_path}") #
        
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist!")
            return {}
        
        documents = {} # dictionary to hold documents
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path) and filename.endswith('.txt'): # Only processes .txt files
                print(f"Processing: {filename}")
                
                try: # Read file content
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    
                    doc_id = str(uuid.uuid4()) # Unique document ID
                    base = os.path.basename(file_path) # Get base filename, base filename is the filename with extension
                    sku = os.path.splitext(base)[0] # Extract SKU from filename, SKU is stored as filename without extension, storage key unit
                    
                    documents[doc_id] = { # Store document with metadata
                        'text': text,
                        'filename': filename,
                        'sku': sku
                    }
                    
                except Exception as e: # Handle file read errors
                    print(f"Error reading {filename}: {e}")
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents): # this function chunks documents into smaller pieces based on chunk_size
        """
        Chunk documents into smaller pieces
        Fixed the regex pattern and chunking logic from the original article
        """
        print("Chunking documents...")
        all_chunks = {} # dictionary to hold all chunks
        
        for doc_id, doc_info in documents.items():
            text = doc_info['text']
            filename = doc_info['filename']
            
            # Fix: Use proper newline pattern
            para_separator = r"\n\n"
            paragraphs = re.split(para_separator, text) # Split by double newlines
            
            for paragraph in paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    words = paragraph.split(" ") # Split paragraph into words
                    
                    current_chunk_str = ""
                    chunks = []
                    
                    for word in words: #` Iterate through words to form chunks
                        # Form new chunk string`
                        if current_chunk_str: # If current chunk is not empty, add space before new word
                            new_chunk = current_chunk_str + " " + word # Add space before new word
                        else:
                            new_chunk = word # Start new chunk with the word
                        
                        # Check if new chunk exceeds token limit
                        if len(self.tokenizer.tokenize(new_chunk)) <= self.chunk_size:
                            current_chunk_str = new_chunk
                        else:
                            if current_chunk_str:
                                chunks.append(current_chunk_str.strip())
                            current_chunk_str = word
                    
                    # Add remaining chunk
                    if current_chunk_str:
                        chunks.append(current_chunk_str.strip())
                    
                    # Create chunk entries
                    for chunk_text in chunks:
                        if chunk_text.strip():  # Only add non-empty chunks
                            chunk_id = str(uuid.uuid4())
                            all_chunks[chunk_id] = {
                                "text": chunk_text,
                                "metadata": {
                                    "doc_id": doc_id,
                                    "file_name": filename
                                }
                            }
        
        print(f"Created {len(all_chunks)} chunks")
        self.documents = all_chunks
        return all_chunks
    
    def create_embeddings(self):
        """Create embeddings for all chunks"""
        print("Creating embeddings...")
        
        texts = []
        chunk_ids = []
        
        for chunk_id, chunk_data in self.documents.items():
            texts.append(chunk_data['text'])
            chunk_ids.append(chunk_id)
        
        # Generate embeddings using sentence-transformers (much easier!)
        embeddings = self.embedding_model.encode(texts)
        
        # Store embeddings
        vector_store = {} # dictionary to hold vector store, text embed into vectors are stored in this dictionary
        
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            vector_store[chunk_id] = embedding.tolist()
        
        self.vector_store = vector_store
        print(f"Created embeddings for {len(vector_store)} chunks")
        return vector_store
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieve most relevant chunks for a query"""
        print(f"Retrieving relevant chunks for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        scores = {} # dictionary to hold similarity scores
        
        # Calculate cosine similarity with all chunks
        for chunk_id, chunk_embedding in self.vector_store.items():
            chunk_embedding = np.array(chunk_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / ( # numerator is what we want to maximize, denominator is the normalization factor
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)  # divided by norms to get cosine similarity, norms is the length of the vector
            )
            
            scores[chunk_id] = similarity # Store similarity score
        
        # Sort by similarity score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k] # Get top_k chunks, why top_k? because we only want the most relevant chunks
        
        # Retrieve full chunk data
        relevant_chunks = []  # list to hold relevant chunks
        for chunk_id, score in sorted_chunks: # Iterate through top_k chunks
            chunk_data = self.documents[chunk_id]
            relevant_chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_data['text'],
                'metadata': chunk_data['metadata'],
                'similarity_score': score
            })
            print(f"Retrieved chunk (score: {score:.4f}): {chunk_data['text'][:100]}...")
        
        return relevant_chunks
    
    def generate_response(self, query, relevant_chunks, model="gpt-3.5-turbo"): 
        """
        Generate response using OpenAI LLM with relevant context
        
        Updated model options:
        - "gpt-3.5-turbo" (cost-effective)
        - "gpt-4" (better quality)
        - "gpt-4-turbo" (latest GPT-4)
        - "gpt-4o" (optimized GPT-4)
        """
        print("Generating LLM response...")
        
        # Combine relevant chunks into context
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Create prompt
        prompt = f"""You are an intelligent assistant. You will be provided with retrieved context and a user's query.
Your job is to understand the request and answer based on the retrieved context.

Here is the context:
<context>
{context}
</context>

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response at this time."
    
    def save_data(self, doc_path="doc_store.json", vector_path="vector_store.json"): # this function saves the document and vector stores to JSON files
        """Save document and vector stores to JSON files"""
        print(f"Saving data to {doc_path} and {vector_path}")
        
        with open(doc_path, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        with open(vector_path, 'w') as f:
            json.dump(self.vector_store, f, indent=2)
    
    def load_data(self, doc_path="doc_store.json", vector_path="vector_store.json"):
        """Load document and vector stores from JSON files"""
        print(f"Loading data from {doc_path} and {vector_path}")
        
        try:
            with open(doc_path, 'r') as f:
                self.documents = json.load(f)
            
            with open(vector_path, 'r') as f:
                self.vector_store = json.load(f)
                
            print(f"Loaded {len(self.documents)} documents and {len(self.vector_store)} embeddings")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def process_documents_pipeline(self, directory_path):
        """Complete pipeline to process documents"""
        print("=== RAG Processing Pipeline ===")
        
        # Step 1: Read documents
        raw_documents = self.read_documents(directory_path)
        
        # Step 2: Chunk documents
        self.chunk_documents(raw_documents)
        
        # Step 3: Create embeddings
        self.create_embeddings()
        
        # Step 4: Save data
        self.save_data()
        
        print("=== Pipeline Complete ===")
    
    def query(self, question, top_k=3):
        """Main query function"""
        print(f"\n=== Processing Query: '{question}' ===")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        # Generate response
        response = self.generate_response(question, relevant_chunks)
        
        return {
            'question': question,
            'response': response,
            'relevant_chunks': relevant_chunks
        }


# Example usage (this will only run if you execute this file directly)
if __name__ == "__main__":
    print("RAG System module loaded successfully!")
    print("Use test_rag.py to test the system or import RAGSystem to use it.")