import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Use SMOTE to balance classes

# Load the transformed dataset
file_path = "transformed_bank_data.csv"  # Ensure this path is correct
df = pd.read_csv(file_path)

# Verify target column 'y' exists and encode it
if "y" not in df.columns:
    raise KeyError("The target column 'y' is not found in the dataset.")
y = df["y"].apply(
    lambda x: 1 if x == "yes" else 0
)  # Encode target (1 for 'yes', 0 for 'no')

# Define features and target
X = df.drop(columns=["y"])  # Exclude the target column 'y' from the features

# Check original class distribution
print("Original Class Distribution in Target:")
print(y.value_counts())

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Verify balanced class distribution
print("\nBalanced Class Distribution:")
print(y_balanced.value_counts())

# Stratified train-test split to maintain class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Verify class distribution in training and testing sets
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTesting set class distribution:")
print(y_test.value_counts())

# Initialize the models
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)

# Train and evaluate Decision Tree model
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
print("Decision Tree Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Train and evaluate Logistic Regression model
logistic_regression.fit(X_train, y_train)
y_pred_logreg = logistic_regression.predict(X_test)
print("\nLogistic Regression Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
