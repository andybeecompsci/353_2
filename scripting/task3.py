# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the transformed dataset
file_path = "transformed_bank_data.csv"  # Ensure this file is correct
df = pd.read_csv(file_path)

# Step 1: Prepare Features and Target
# Encode the target column if not already binary
if df["y"].dtype == "object":  # Ensuring 'y' is encoded
    df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)

# Convert y to integer type explicitly to avoid continuous label issues
df["y"] = df["y"].astype(int)

# Verify the contents and data type of the target column
print("Target Variable 'y' Distribution After Encoding:")
print(df["y"].value_counts())
print("Data type of 'y':", df["y"].dtype)

# Define features (X) and target (y)
X = df.drop(columns=["y"])
y = df["y"]

# Step 2: Train-Test Split with Stratification to handle imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verify the class distribution after splitting
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTesting set class distribution:")
print(y_test.value_counts())

# Step 3: Apply SMOTE only if y_train is imbalanced and contains both classes
if y_train.nunique() > 1:  # Check if both classes are present
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("\nBalanced Training set class distribution after SMOTE:")
    print(y_train_balanced.value_counts())
else:
    # Use original training set if SMOTE can't be applied
    X_train_balanced, y_train_balanced = X_train, y_train
    print("SMOTE not applied due to class imbalance issue.")

# Step 4: Initialize Models
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
random_forest = RandomForestClassifier(random_state=42)

# Step 5: Train and Evaluate Models

# Decision Tree Model
print("\nDecision Tree Model Evaluation:")
decision_tree.fit(X_train_balanced, y_train_balanced)
y_pred_tree = decision_tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Logistic Regression Model
print("\nLogistic Regression Model Evaluation:")
logistic_regression.fit(X_train_balanced, y_train_balanced)
y_pred_logreg = logistic_regression.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# Random Forest (Ensemble) Model
print("\nRandom Forest Model (Ensemble) Evaluation:")
random_forest.fit(X_train_balanced, y_train_balanced)
y_pred_rf = random_forest.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
