import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PROFILING TIMEEEEE


# load the dataset
file_path = "bank-additional-full.csv"
df = pd.read_csv(file_path, sep=";")

# basic overview
print("data overview:")
print(df.info())
print("\nfirst 5 rows of the dataset:")
print(df.head())

# summary stats
print("\nstatistical summary of numerical features:")
print(df.describe())

# check for missing values
print("\nmissing values:")
missing_values = df.isin(["unknown"]).sum()
print(missing_values[missing_values > 0])

# missing values percentage
# print("\nmissing values %:")
# missing_percentage = (df.isin(["unknown"]).sum() / len(df)) * 100
# print(missing_percentage[missing_percentage > 0])


# categorical variable analysis
print("\nvalue counts for categorical variables:")
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# target variable distribution
print("\ntarget variable 'y' distribution:")
print(df["y"].value_counts())

# initial visualizations
# numerical features
df.hist(column=["age", "campaign", "pdays", "previous"], bins=20, figsize=(10, 8))
plt.suptitle("distribution of numerical features")
plt.show()

# categorical feature bar plots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
sns.countplot(data=df, x="job", ax=axes[0, 0])
sns.countplot(data=df, x="marital", ax=axes[0, 1])
sns.countplot(data=df, x="education", ax=axes[1, 0])
sns.countplot(data=df, x="contact", ax=axes[1, 1])
sns.countplot(data=df, x="month", ax=axes[2, 0])
sns.countplot(data=df, x="day_of_week", ax=axes[2, 1])
plt.tight_layout()
plt.show()

# correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm"
)
plt.title("correlation matrix for numerical features")
plt.show()
