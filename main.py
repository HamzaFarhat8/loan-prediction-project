

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv('loan.csv')
print("Initial Data:\n", data.head())
print("\nColumns:", data.columns)

# -----------------------------
# Handle missing values
# -----------------------------
# Fill categorical columns with mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Fill numerical columns with median
for col in ['LoanAmount', 'Loan_Amount_Term']:
    data[col].fillna(data[col].median(), inplace=True)

# -----------------------------
# Encode target variable
# -----------------------------
le = LabelEncoder()
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])  # Y: 1=Approved, 0=Rejected

# -----------------------------
# Train-Test Split
# -----------------------------
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

# Encode categorical features
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Feature Scaling (for LR & XGB)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Model Comparison
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

print("\nModel Comparison Results:\n")

for name, model in models.items():
    # Use scaled data for Logistic Regression & XGBoost
    if name in ['Logistic Regression', 'XGBoost']:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds))
    
    # Feature importance for tree-based models
    if name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
            feat_imp = feat_imp.sort_values(ascending=False)
            plt.figure(figsize=(10,5))
            sns.barplot(x=feat_imp[:10], y=feat_imp[:10].index)
            plt.title(f'Top 10 Feature Importance - {name}')
            plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
