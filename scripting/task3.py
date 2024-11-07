import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the transformed dataset
file_path = "transformed_bank_data.csv"  # Ensure this path is correct
df = pd.read_csv(file_path)

# Verify and Encode Target as Binary
# Checking if y contains only binary values
print("Target Variable 'y' Distribution After Transformation:")
print(df["y"].value_counts())

# Ensure target column y is binary (0 and 1)
df["y"] = df["y"].apply(lambda x: 1 if x == "yes" or x == 1 else 0)

# Define features and target
X = df.drop(columns=["y"])
y = df["y"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the models
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
random_forest = RandomForestClassifier(random_state=42)

# Train and evaluate Decision Tree model
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
print("Decision Tree Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Train and evaluate Logistic Regression model
logistic_regression.fit(X_train, y_train)
y_pred_logreg = logistic_regression.predict(X_test)
print("\nLogistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# Train and evaluate Random Forest (Ensemble) model
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
print("\nRandom Forest Model (Ensemble) Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
