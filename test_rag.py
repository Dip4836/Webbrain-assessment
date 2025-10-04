import os
from agent.rag_agent import RAGAgent

print("Current directory:", os.getcwd())
print("Files in current directory:")
for item in os.listdir('.'):
    print(f"  {item}")


if os.path.exists('docs'):
    print("\nFiles in docs folder:")
    for root, dirs, files in os.walk('docs'):
        for file in files:
            print(f"  {os.path.join(root, file)}")
else:
    print("\nNo 'docs' folder found!")


if os.path.exists('rag_eval_questions.csv'):
    print("\n Found rag_eval_questions.csv")
    import pandas as pd
    df = pd.read_csv('rag_eval_questions.csv')
    print(f"Questions file has {len(df)} questions")
    print("Sample questions:")
    for i, row in df.head(3).iterrows():
        print(f"  {row['id']}: {row['question'][:50]}...")
else:
    print("\n rag_eval_questions.csv not found!")
    print("Creating a dummy questions file for testing...")


    dummy_questions = {
        'id': [1, 2, 3],
        'question': [
            'What is the main topic discussed in the documents?',
            'How can we improve customer retention?',
            'What are the key business metrics mentioned?'
        ]
    }
    import pandas as pd
    pd.DataFrame(dummy_questions).to_csv('rag_eval_questions.csv', index=False)
    print(" Created dummy rag_eval_questions.csv")