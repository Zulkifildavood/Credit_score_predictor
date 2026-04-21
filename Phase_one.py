import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_auc_score



print("1. Loading Real Financial Data...")

file_path = "loan_approval_dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

df = pd.read_csv(file_path)

print(f"Initial dataset shape: {df.shape}")

print("2. Cleaning and Encoding Data...")

# --- Normalize column names ---
df.columns = df.columns.str.strip().str.lower()

# --- Drop unnecessary column ---
if 'loan_id' in df.columns:
    df.drop(columns=['loan_id'], inplace=True)

# --- Normalize ALL string values ---
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# --- Check if target exists ---
if 'loan_status' not in df.columns:
    raise ValueError("❌ Target column 'loan_status' is missing!")

# --- Debug before mapping ---
print("\nUnique values BEFORE mapping:")
print(df['loan_status'].value_counts())

# --- Encode target ---
df['loan_status'] = df['loan_status'].map({
    'approved': 1,
    'rejected': 0
})

# --- Debug after mapping ---
print("\nUnique values AFTER mapping:")
print(df['loan_status'].value_counts(dropna=False))

# --- Drop rows where mapping failed ---
df.dropna(subset=['loan_status'], inplace=True)

print(f"Rows after target cleaning: {len(df)}")

# --- Encode categorical features safely ---
categorical_mappings = {
    'education': {'graduate': 1, 'not graduate': 0},
    'self_employed': {'yes': 1, 'no': 0}
}

for col, mapping in categorical_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# --- Handle missing values ---
# --- Handle missing values properly ---

# Numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns (after mapping)
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("unknown")

# --- Final safety check ---
if len(df) == 0:
    raise ValueError("❌ Dataset became empty after preprocessing. Check mappings!")

print(f"\nFinal dataset shape: {df.shape}")

print("3. Defining Features and Splitting Data...")

X = df.drop(columns=['loan_status'])
y = df['loan_status']

print("\n📊 Target distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# --- Stratified split (important for classification) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("4. Applying Feature Scaling...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("5. Training the Logistic Regression Model...")

model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'   
)

model.fit(X_train_scaled, y_train)

# --- Predictions ---
# Get probabilities instead
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Prevent extreme 0% / 100% confidence
y_prob = np.clip(y_prob, 0.01, 0.99)

# Custom threshold (tune this: 0.4–0.7)
threshold = 0.6

y_pred = (y_prob >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Performance Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Rejected (0)", "Approved (1)"]
))

print("6. Saving the Model and Scaler...")

with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


print("\n🧱 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🎯 ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

print("\n✅ Saved 'credit_model.pkl' and 'scaler.pkl'. Ready for Streamlit/AWS!")