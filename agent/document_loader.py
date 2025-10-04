import os
import json
from pathlib import Path
import pytesseract
from PIL import Image

class DocumentLoader:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder
        self.documents = []
    
    def load_all_documents(self):
        """Load all documents from the docs folder"""
        docs_path = Path(self.docs_folder)
        
        for file_path in docs_path.rglob('*'):
            if file_path.is_file():
                content = self._load_file(file_path)
                if content:
                    self.documents.append({
                        'filename': file_path.name,
                        'content': content,
                        'filepath': str(file_path)
                    })
        
        print(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def _load_file(self, file_path):
        """Load content from different file types"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.md':
                return self._load_markdown(file_path)
            elif suffix in ['.png', '.jpg', '.jpeg']:
                return self._load_image_ocr(file_path)
            elif suffix == '.txt':
                return self._load_text(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _load_markdown(self, file_path):
        """Load markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_text(self, file_path):
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_image_ocr(self, file_path):
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"OCR failed for {file_path}: {e}")
            return None