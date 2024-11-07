import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the original dataset
file_path = "bank-additional-full.csv"
df = pd.read_csv(file_path, sep=";")

# Separate features (X) and target variable (y)
y = df["y"].apply(lambda x: 1 if x == "yes" else 0)  # Convert target to binary
X = df.drop(columns=["y"])

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Create transformers for numerical and categorical data
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
        ("scaler", StandardScaler()),  # Standardize numerical data
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="missing"),
        ),  # Fill missing with 'missing'
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),  # One-hot encode categorical data
    ]
)

# Combine transformers into a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Apply transformations to the features
X_transformed = preprocessor.fit_transform(X)
feature_names = (
    preprocessor.get_feature_names_out()
)  # Get feature names after transformation

# Convert transformed features to DataFrame
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
X_transformed_df["y"] = y  # Add target variable back

# Save the new dataset with transformed features
X_transformed_df.to_csv("transformed_bank_data.csv", index=False)
print("Task 2 complete. Transformed data saved to 'transformed_bank_data.csv'.")
