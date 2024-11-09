import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


# MACHINE LEARNING TIMEEEEEEEEEE

# load the transformed dataset
file_path = "transformed_bank_data.csv"  # ensure this file is correct
df = pd.read_csv(file_path)

# prepare features and target
# encode the target column if not already binary
if df["y"].dtype == "object":  # ensuring 'y' is encoded
    df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)

# convert y to integer type explicitly to avoid continuous label issues
df["y"] = df["y"].astype(int)

# verify the contents and data type of the target column
print("target variable 'y' distribution after encoding:")
print(df["y"].value_counts())
print("data type of 'y':", df["y"].dtype)

# define features (x) and target (y)
X = df.drop(columns=["y"])
y = df["y"]

# train-test split with strat to deal w imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# verify class distribution after splitting
print("\ntraining set class distribution:")
print(y_train.value_counts())
print("\ntesting set class distribution:")
print(y_test.value_counts())

# apply smote only if y_train is imbalanced and contains both classes
if y_train.nunique() > 1:  # check if both classes exist
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("\nbalanced training set class distribution after smote:")
    print(y_train_balanced.value_counts())
else:
    # use original training set if smote cant be applied
    X_train_balanced, y_train_balanced = X_train, y_train
    print("smote not applied due to class imbalance issue.")

# initialize models
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
random_forest = RandomForestClassifier(random_state=42)

# train and finish models
# decision tree model
print("\ndecision tree model evaluation:")
decision_tree.fit(X_train_balanced, y_train_balanced)
y_pred_tree = decision_tree.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# log reg model
print("\nlogistic regression model evaluation:")
logistic_regression.fit(X_train_balanced, y_train_balanced)
y_pred_logreg = logistic_regression.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# random forest model
print("\nrandom forest model (ensemble) evaluation:")
random_forest.fit(X_train_balanced, y_train_balanced)
y_pred_rf = random_forest.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
