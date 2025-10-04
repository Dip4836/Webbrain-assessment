
import json, os, csv

def grade(answers_path: str):
    with open(answers_path, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    with open(os.path.join(os.path.dirname(answers_path),'rag_eval_questions.csv'), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    score = 0
    results = []
    for row in rows:
        qid = int(row['qid'])
        expected = row['expected_contains'].lower()
        got = (answers.get(str(qid), "") or "").lower()
        passed = expected in got
        results.append({"qid": qid, "pass": passed, "expected": expected, "got": got[:120]})
        score += int(passed)
    return {"score": score, "total": len(rows), "results": results}

if __name__ == "__main__":
    import sys, json
    out = grade(sys.argv[1])
    print(json.dumps(out, indent=2))
