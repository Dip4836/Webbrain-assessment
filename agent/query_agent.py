# query_agent.py
import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List

# Config
EMB_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "agent/faiss.index"
METADATA_PATH = "agent/metadata.json"
QUESTIONS_PATHS = ["rag_eval_questions.csv", "rag_eval_questions.xlsx"]  
OUTPUT = "answers.json"
TOPK = 4
MAX_CHARS_PER_EVIDENCE = 800 


def load_index_and_meta(index_path=INDEX_PATH, meta_path=METADATA_PATH):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found. Run build_index.py first.")
    idx = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return idx, metadata


def load_questions():
    for p in QUESTIONS_PATHS:
        if os.path.exists(p):
            if p.endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)

            if "qid" in df.columns and "question" in df.columns:
                return df[["qid", "question"]].to_dict(orient="records")
            elif "id" in df.columns and "question" in df.columns:
                return df[["id", "question"]].rename(columns={"id": "qid"}).to_dict(orient="records")

            cols = list(df.columns)
            return [{"qid": str(r[cols[0]]), "question": str(r[cols[1]])} for _, r in df.iterrows()]
    raise FileNotFoundError("rag_eval_questions file not found (csv/xlsx).")

# Retrieval
def retrieve(query: str, idx, metadata, embed_model: SentenceTransformer, topk=TOPK):
    q_emb = embed_model.encode(query).astype("float32")

    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = idx.search(q_emb.reshape(1, -1), topk)
    hits = []
    for i in I[0]:
        if i < 0 or i >= len(metadata):
            continue
        meta = metadata[i]
        hits.append(meta)
    return hits

def compose_answer(question: str, hits: List[dict]):
    pieces = []
    files = []
    for h in hits:
        txt = h.get("text", "")
        files.append(Path(h.get("path", "")).name)

        pieces.append(txt.strip().replace("\n", " ")[:MAX_CHARS_PER_EVIDENCE])
    if len(pieces) == 0:
        body = "No relevant document found."
    else:
        body = " ".join(pieces)

    sources = ", ".join(sorted(set(files)))

    answer = f"{body}\n\nSources: {sources}"
    return answer

def main():
    print("Loading index and metadata...")
    idx, metadata = load_index_and_meta()
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMB_MODEL)

    print("Loading questions...")
    questions = load_questions()
    print(f"Found {len(questions)} questions.")

    answers = {}
    for qa in questions:
        qid = str(qa["qid"])  
        qtext = str(qa["question"])
        print(f"Processing {qid}: {qtext[:80]}...")
        hits = retrieve(qtext, idx, metadata, embed_model, topk=TOPK)
        ans = compose_answer(qtext, hits)
      
        answers[qid] = ans

   
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT} with {len(answers)} answers.")

if __name__ == "__main__":
    main()