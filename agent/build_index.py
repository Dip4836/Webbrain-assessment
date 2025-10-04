# agent/build_index.py - 
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
from ingest import load_docs

# Configuration
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300 
CHUNK_OVERLAP = 50
INDEX_PATH = "agent/faiss.index"
METADATA_PATH = "agent/metadata.json"

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    words = text.split()
    if len(words) <= size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk_words = words[i:i + size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
     
        if i + size >= len(words):
            break
    
    return chunks

def build_index():
    """Build FAISS index from documents"""
    print("Loading documents...")
    docs = load_docs("docs")
    
    if not docs:
        print("No documents found! Make sure docs/ folder exists with content.")
        return
    
    print("Loading embedding model...")
    model = SentenceTransformer(EMB_MODEL)
    
    embeddings = []
    metadata = []
    
    print("Processing documents and creating chunks...")
    for doc in tqdm(docs, desc="Processing docs"):
        chunks = chunk_text(doc['text'])
        print(f"Document {doc['filename']}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            if chunk.strip(): 
              
                emb = model.encode(chunk)
                embeddings.append(emb)
                
                
                metadata.append({
                    "path": doc['path'],
                    "filename": doc['filename'],
                    "chunk_id": i,
                    "text": chunk[:1000]  
                })
    
    if not embeddings:
        print("No valid chunks created!")
        return
    
    print(f"Created {len(embeddings)} embeddings")
    
    # Create FAISS index
    print("Building FAISS index...")
    emb_matrix = np.vstack(embeddings).astype('float32')
    
  
    faiss.normalize_L2(emb_matrix)
    
    # Create index
    index = faiss.IndexFlatIP(emb_matrix.shape[1]) 
    index.add(emb_matrix)
    
    
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Total chunks indexed: {len(metadata)}")
    
    
    print("\nSample chunks:")
    for i, meta in enumerate(metadata[:3]):
        print(f"{i+1}. File: {meta['filename']}")
        print(f"   Text: {meta['text'][:100]}...")
        print()

if __name__ == "__main__":
    build_index()