# Setup and Run Instructions

## Prerequisites
- Python 3.8+
- pip package manager
- Tesseract OCR (for image text extraction)

## Installation

### 1. Install Tesseract OCR
**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Or using chocolatey:
choco install tesseract
```

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Running the Analysis

### Part A: Churn Analysis
```bash
# Run the churn analysis
python analysis.py
```

**Outputs:**
- `figs/analysis_overview.png` - Dataset and model overview
- `figs/feature_importance.png` - Top predictive features
- `feature_importances.csv` - Complete feature rankings
- `models/lgbm_churn_model.pkl` - Trained model
- `models/label_encoders.pkl` - Preprocessing encoders

### Part B: RAG Agent

#### Step 1: Build the Document Index
```bash
# Process documents and build FAISS index
cd agent
python build_index.py
```

**Outputs:**
- `agent/faiss.index` - Vector search index
- `agent/metadata.json` - Document chunk metadata

#### Step 2: Run Q&A Agent
```bash
# Answer evaluation questions
python query_agent.py
```

**Outputs:**
- `answers.json` - Q&A results with citations

#### Step 3: Evaluate Results
```bash
# Run the grader
python eval_stub.py answers.json
```

## Project Structure
```
├── analysis.py              # Churn analysis script
├── churn_533064950.csv      # Churn dataset
├── rag_eval_questions.csv   # Evaluation questions
├── eval_stub.py             # Auto-grader
├── requirements.txt         # Python dependencies
├── INSIGHTS.md             # Analysis insights
├── RUN.md                  # This file
├── docs/                   # Documentation corpus
│   ├── *.md               # Markdown files
│   └── *.png              # OCR images
├── agent/                  # RAG system
│   ├── ingest.py          # Document loading
│   ├── build_index.py     # Index creation
│   └── query_agent.py     # Q&A system
├── figs/                   # Generated plots
├── models/                 # Saved models
└── answers.json           # Final Q&A results
```

## Troubleshooting

**Tesseract Issues:**
- Ensure tesseract is in your PATH
- On Windows, you may need to set: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

**Memory Issues:**
- Reduce CHUNK_SIZE in build_index.py if running on limited memory
- Consider using faiss-gpu for large document collections

**Import Errors:**
- Verify all requirements are installed: `pip install -r requirements.txt`
- Use virtual environment if experiencing conflicts
