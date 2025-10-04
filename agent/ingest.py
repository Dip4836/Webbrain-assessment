# agent/ingest.py - Document ingestion with OCR support
import os
import glob
from PIL import Image
import pytesseract
import mimetypes
from pathlib import Path

def extract_text_from_file(path):
    """Extract text from various file types including images with OCR"""
    try:
     
        file_ext = Path(path).suffix.lower()
        
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            
            print(f"Performing OCR on {path}")
            image = Image.open(path)
            text = pytesseract.image_to_string(image)
            return text
        elif file_ext in ['.md', '.txt']:
           
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return text
        else:
           
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return text
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return ""

def load_docs(folder="docs"):
    """Load all documents from the specified folder"""
    docs = []
    
    if not os.path.exists(folder):
        print(f"Warning: {folder} directory not found!")
        return docs
    
   
    pattern = os.path.join(folder, "**", "*")
    all_files = glob.glob(pattern, recursive=True)
    
    for file_path in all_files:
        if os.path.isfile(file_path):
            print(f"Processing: {file_path}")
            text = extract_text_from_file(file_path)
            if text.strip():  
                docs.append({
                    "path": file_path,
                    "text": text,
                    "filename": Path(file_path).name
                })
                print(f"  -> Extracted {len(text)} characters")
            else:
                print(f"  -> No text extracted")
    
    print(f"\nSuccessfully loaded {len(docs)} documents")
    return docs

if __name__ == "__main__":

    docs = load_docs("docs")
    

    for doc in docs:
        print(f"\nFile: {doc['filename']}")
        print(f"Path: {doc['path']}")
        print(f"Text preview: {doc['text'][:200]}...")
        
    print(f"\nTotal documents processed: {len(docs)}")