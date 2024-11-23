import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Read the dataset
df = pd.read_csv("cervicaldata.csv")

# Replace "?" with NaN
df = df.replace("?", np.nan)

# Drop the specific columns
df = df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"])

# Convert all columns to numeric values, coercing errors (non-numeric values will be turned to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# List of columns to exclude from filling NaN with mean
exclude_columns = [col for col in df.columns if col.startswith('Dx')]

# Fill NaN values with the mean of columns (excluding 'Dx' columns)
for col in df.columns:
    if col not in exclude_columns:
        # Calculate the mean excluding NaN values
        avg_value = df[col].dropna().mean()
        df[col] = df[col].fillna(avg_value)  # Fill NaN with average value

# Display the first 20 rows of the DataFrame
print(df.head(20))

# Select all columns except the target column for features
features = [col for col in df.columns if col != 'Dx:Cancer']
target = 'Dx:Cancer'

# Split the data for training and testing
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a list to store the accuracy scores
accuracy_scores = []

# Decision Tree
print("### Decision Tree ###")
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_scores.append(accuracy_dt)  # Store the accuracy
print(f"Decision Tree Accuracy = {accuracy_dt:.4f}\n")

# Random Forest
print("### Random Forest ###")
random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_scores.append(accuracy_rf)  # Store the accuracy
print(f"Random Forest Accuracy = {accuracy_rf:.4f}\n")

# Support Vector Machine
print("### Support Vector Machine ###")
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_scores.append(accuracy_svm)  # Store the accuracy
print(f"SVM Accuracy = {accuracy_svm:.4f}\n")

# Logistic Regression
print("### Logistic Regression ###")
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_scores.append(accuracy_lr)  # Store the accuracy
print(f"Logistic Regression Accuracy = {accuracy_lr:.4f}\n")

# K-Nearest Neighbors
print("### K-Nearest Neighbors ###")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_scores.append(accuracy_knn)  # Store the accuracy
print(f"KNN Accuracy = {accuracy_knn:.4f}\n")

# Gradient Boosting
print("### Gradient Boosting ###")
gradient_boosting = GradientBoostingClassifier(random_state=42)
gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
accuracy_scores.append(accuracy_gb)  # Store the accuracy
print(f"Gradient Boosting Accuracy = {accuracy_gb:.4f}\n")

# Calculate the average accuracy
average_accuracy = np.mean(accuracy_scores)
print(f"Average Accuracy of all models = {average_accuracy:.4f}")
