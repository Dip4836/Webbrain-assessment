# analysis.py 
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    import lightgbm as lgb
    import joblib
    import os  
    import warnings
    warnings.filterwarnings('ignore')
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install missing packages with: pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib")
    exit(1)


import sys
print(f"Python version: {sys.version}")


packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'lightgbm', 'joblib']
for pkg in packages:
    try:
        __import__(pkg)
        print(f" {pkg} - OK")
    except ImportError:
        print(f" {pkg} - MISSING")

# Ensure directories exist
import os
os.makedirs('figs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Check for data files
os.chdir(r'c:\Webbains')
files = ['churn_533064950.csv', 'churn.csv']
for file in files:
    if os.path.exists(file):
        print(f"Found data file: {file}")
    else:
        print(f"Data file not found: {file}")

def load_and_explore_data():
    """Load and perform initial exploration of the churn dataset"""
  
    data_files = ["churn_533064950.csv", "churn.csv"]
    df = None
    
    for file in data_files:
        if os.path.exists(file):
            print(f"Loading dataset: {file}")
            df = pd.read_csv(file)
            break
    
    if df is None:
        raise FileNotFoundError("No churn dataset found!")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    target_candidates = ['churn', 'Churn', 'CHURN', 'churned', 'Churned', 'label_churned']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print("Available columns:", df.columns.tolist())
       
        churn_cols = [col for col in df.columns if 'churn' in col.lower()]
        if churn_cols:
            target_col = churn_cols[0]
            print(f"Found churn-related column: {target_col}")
        else:
            target_col = df.columns[-1]
            print(f"Using last column as target: {target_col}")
    
    print(f"Target column: {target_col}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    
    unique_vals = df[target_col].nunique()
    if unique_vals > 2:
        print(f"WARNING: Target has {unique_vals} classes, expected binary for churn analysis")
        if 'label_churned' in df.columns:
            print("Using 'label_churned' instead as it appears to be the actual churn target")
            target_col = 'label_churned'
            print(f"New target distribution:\n{df[target_col].value_counts()}")
    
    if target_col in df.columns:
        print(f"Class imbalance ratio: {df[target_col].value_counts().min() / df[target_col].value_counts().max():.3f}")
    
    return df, target_col

def preprocess_data(df, target_col):
    """Preprocess the data for modeling"""
   
    print(f"Missing values:\n{df.isnull().sum()}")
    
   
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
   
    categorical_columns = X.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_columns)}")
    

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    
   
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    return X, y, label_encoders

def create_visualizations(df, target_col, y_test, y_scores):
    """Create and save visualizations"""
    plt.style.use('default')
    
   
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    df[target_col].value_counts().plot(kind='bar')
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    
   
    plt.subplot(1, 3, 2)
    prec, rec, _ = precision_recall_curve(y_test, y_scores)
    auc_pr = auc(rec, prec)
    plt.plot(rec, prec, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, 'Feature Importance\n(see separate plot)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('figs/analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return auc_pr

print("=== CHURN ANALYSIS STARTED ===")
df, target_col = load_and_explore_data()

X, y, label_encoders = preprocess_data(df, target_col)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training set churn rate: {y_train.mean():.3f}")
print(f"Test set churn rate: {y_test.mean():.3f}")


print("\n=== TRAINING MODEL ===")
clf = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    class_weight='balanced', 
    random_state=42,
    early_stopping_rounds=50,
    verbose=-1
)


clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc'
)


y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)


prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
auc_pr = auc(rec, prec)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"AUC-PR: {auc_pr:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== TOP 10 MOST IMPORTANT FEATURES ===")
print(feature_importance.head(10))


auc_pr_vis = create_visualizations(df, target_col, y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


feature_importance.to_csv('feature_importances.csv', index=False)
joblib.dump(clf, 'models/lgbm_churn_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print(f"\n=== BUSINESS INSIGHTS ===")
print("Based on the analysis:")
print(f"1. Model achieves AUC-PR of {auc_pr:.3f}, indicating {'good' if auc_pr > 0.6 else 'moderate'} predictive performance")
print("2. Top predictive features for churn:")
for i, row in feature_importance.head(5).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.3f}")

print(f"\n=== BUSINESS RECOMMENDATION ===")
print("Recommendation: Focus retention efforts on customers with high-risk profiles")
print("identified by the top predictive features. Implement targeted interventions")
print("for customers scoring above the 80th percentile in churn probability.")

print(f"\n=== FILES SAVED ===")
print("- figs/analysis_overview.png")
print("- figs/feature_importance.png") 
print("- feature_importances.csv")
print("- models/lgbm_churn_model.pkl")
print("- models/label_encoders.pkl")