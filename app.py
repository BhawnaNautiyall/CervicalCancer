from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Read and preprocess the dataset
df = pd.read_csv("cervicaldata.csv")
df = df.replace("?", np.nan)
df = df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"])
df = df.apply(pd.to_numeric, errors='coerce')
exclude_columns = [col for col in df.columns if col.startswith('Dx')]
for col in df.columns:
    if col not in exclude_columns:
        avg_value = df[col].dropna().mean()
        df[col] = df[col].fillna(avg_value)

# Split dataset
features = [col for col in df.columns if col != 'Dx:Cancer']
target = 'Dx:Cancer'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
decision_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
random_forest = RandomForestClassifier(random_state=42, n_estimators=100).fit(X_train, y_train)
svm = SVC(random_state=42, probability=True).fit(X_train, y_train)  
logistic_regression = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
gradient_boosting = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html', features=features)

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Prepare input data
        input_data = []
        for feature in features:
            value = request.form.get(feature, "")
            if value == "":
                # Use column average if input is empty
                avg_value = df[feature].mean()
                input_data.append(avg_value)
            else:
                input_data.append(float(value))  # Convert to float

        # Reshape for prediction
        user_input = np.array(input_data).reshape(1, -1)

        # Predictions with probabilities
        predictions = {
            "Decision Tree": f"{decision_tree.predict_proba(user_input)[0][1] * 100:.2f}%", 
            "Random Forest": f"{random_forest.predict_proba(user_input)[0][1] * 100:.2f}%",
            "SVM": f"{svm.predict_proba(user_input)[0][1] * 100:.2f}%",  
            "Logistic Regression": f"{logistic_regression.predict_proba(user_input)[0][1] * 100:.2f}%",
            "KNN": f"{knn.predict_proba(user_input)[0][1] * 100:.2f}%",
            "Gradient Boosting": f"{gradient_boosting.predict_proba(user_input)[0][1] * 100:.2f}%"
        }

        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
