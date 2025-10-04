import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from document_loader import DocumentLoader

class RAGAgent:
    def __init__(self, docs_folder, model_name="all-MiniLM-L6-v2"):
        self.docs_folder = docs_folder
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def setup(self):
        """Initialize the RAG system"""
        print("Setting up RAG agent...")
        
        # Load documents
        loader = DocumentLoader(self.docs_folder)
        self.documents = loader.load_all_documents()
        
        # Create chunks
        self._create_chunks()
        
        # Create embeddings and index
        self._create_embeddings()
        self._create_index()
        
        print("RAG agent setup complete!")
    
    def _create_chunks(self, chunk_size=500):
        """Split documents into chunks"""
        self.chunks = []
        
        for doc in self.documents:
            content = doc['content']
            filename = doc['filename']
            
            # Simple chunking by character count
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                if len(chunk.strip()) > 50:  # Skip very short chunks
                    self.chunks.append({
                        'text': chunk.strip(),
                        'filename': filename,
                        'chunk_id': len(self.chunks)
                    })
        
        print(f"Created {len(self.chunks)} chunks")
    
    def _create_embeddings(self):
        """Create embeddings for all chunks"""
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.model.encode(chunk_texts)
        print(f"Created embeddings: {self.embeddings.shape}")
    
    def _create_index(self):
        """Create FAISS index for similarity search"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def retrieve(self, query, top_k=3):
        """Retrieve relevant chunks for a query"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'filename': chunk['filename'],
                    'score': float(score)
                })
        
        return results
    
    def answer_question(self, question):
        """Answer a question using retrieved context"""
        # Retrieve relevant chunks
        context_chunks = self.retrieve(question, top_k=3)
        
        # Combine context
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        source_files = list(set([chunk['filename'] for chunk in context_chunks]))
        
        # Simple rule-based answering (replace with LLM if available)
        answer = self._generate_answer(question, context)
        
        return {
            'answer': answer,
            'sources': source_files,
            'context': context[:500] + "..." if len(context) > 500 else context
        }
    
    def _generate_answer(self, question, context):
        """Generate answer based on context (simplified version)"""
        # This is a simplified approach - in practice, you'd use an LLM
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Simple keyword matching and extraction
        if "what" in question_lower:
            # Look for definitions or explanations
            sentences = context.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_lower.split()):
                    return sentence.strip()
        
        elif "how" in question_lower:
            # Look for process descriptions
            sentences = context.split('.')
            for sentence in sentences:
                if "step" in sentence.lower() or "process" in sentence.lower():
                    return sentence.strip()
        
        # Fallback: return first relevant sentence
        sentences = context.split('.')[:3]
        return '. '.join(sentences).strip()

def process_evaluation_questions(agent, questions_file, output_file):
    """Process evaluation questions and save answers"""
    # Load questions
    questions_df = pd.read_csv(questions_file)
    
    answers = {}
    
    for _, row in questions_df.iterrows():
        question_id = row['id']
        question = row['question']
        
        print(f"Processing question {question_id}: {question[:50]}...")
        
        result = agent.answer_question(question)
        
        answers[str(question_id)] = {
            'answer': result['answer'],
            'sources': result['sources']
        }
    
    # Save answers
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=2)
    
    print(f"Saved answers to {output_file}")
    return answers

if __name__ == "__main__":
    # Initialize and run RAG agent
    docs_folder = "../docs"  # Adjust path as needed
    questions_file = "../rag_eval_questions.csv"
    output_file = "../answers.json"
    
    agent = RAGAgent(docs_folder)
    agent.setup()
    
    # Process evaluation questions
    answers = process_evaluation_questions(agent, questions_file, output_file)
    
    print("RAG evaluation complete!")