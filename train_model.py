import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print("1. Generating Mock Data...")
np.random.seed(42) # For reproducibility

# Generate 5000 mock customers
num_records = 5000

# Realistic-ish financial distributions
income = np.random.normal(60000, 20000, num_records).clip(20000, 200000)
loan_amount = np.random.normal(20000, 10000, num_records).clip(5000, 100000)
credit_utilization = np.random.uniform(0.1, 0.95, num_records)
# 1 = Good history, 0 = Bad history (80% of people have good history)
repayment_history = np.random.choice([1, 0], size=num_records, p=[0.8, 0.2])

# Create DataFrame
df = pd.DataFrame({
    'income': income,
    'loan_amount': loan_amount,
    'credit_utilization': credit_utilization,
    'repayment_history': repayment_history
})

# Create the target variable (Credit Label: 1 = Good Risk, 0 = Bad Risk)
# We invent a simple rule for the mock data: High utilization and bad history = High Risk
risk_score = (df['income'] / df['loan_amount']) + (df['repayment_history'] * 3) - (df['credit_utilization'] * 2)
df['credit_label'] = np.where(risk_score > 3.0, 1, 0) # 1 is approved, 0 is denied

# Save the mock data to a CSV so you can use it later for batch testing
df.to_csv("mock_financial_data.csv", index=False)
print("Mock data saved as 'mock_financial_data.csv'")

print("2. Training the Logistic Regression Model...")
# Define features (X) and target (y)
X = df[['income', 'loan_amount', 'credit_utilization', 'repayment_history']]
y = df['credit_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check quick accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully! Accuracy on test data: {accuracy * 100:.2f}%")

print("3. Saving the Model...")
# Save the model to a pickle file
with open('credit_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'credit_model.pkl'. Ready for inference!")