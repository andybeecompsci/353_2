import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "bank-additional-full.csv"
df = pd.read_csv(file_path, sep=";")

# Basic Overview
print("Data Overview:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Summary Stats
print("\nStatistical Summary of Numerical Features:")
print(df.describe())

# Check for Missing Values
print("\nMissing Values:")
missing_values = df.isin(["unknown"]).sum()
print(missing_values[missing_values > 0])

# Categorical Variable Analysis
print("\nValue Counts for Categorical Variables:")
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# Target Variable Distribution
print("\nTarget Variable 'y' Distribution:")
print(df["y"].value_counts())

# Initial Visualizations
# Numerical features
df.hist(column=["age", "campaign", "pdays", "previous"], bins=20, figsize=(10, 8))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Categorical feature bar plots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
sns.countplot(data=df, x="job", ax=axes[0, 0])
sns.countplot(data=df, x="marital", ax=axes[0, 1])
sns.countplot(data=df, x="education", ax=axes[1, 0])
sns.countplot(data=df, x="contact", ax=axes[1, 1])
sns.countplot(data=df, x="month", ax=axes[2, 0])
sns.countplot(data=df, x="day_of_week", ax=axes[2, 1])
plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm"
)
plt.title("Correlation Matrix for Numerical Features")
plt.show()
