import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the original dataset
file_path = "bank-additional-full.csv"
df = pd.read_csv(file_path, sep=";")

# Step 1: Handle Missing or Unknown Values
# Replace "unknown" with NaN to simplify handling of missing values
df = df.replace("unknown", pd.NA)

# Fill missing values in categorical columns with the mode
for column in df.select_dtypes(include=["object"]).columns:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)

# Step 2: Encode Categorical Variables
# Binary Encoding for the Target Variable 'y'
df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)

# One-Hot Encoding for other categorical variables
df = pd.get_dummies(df, drop_first=True)

# Step 3: Scale Numerical Features
# Select only numerical columns for scaling
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 4: Verify Target Variable's Class Distribution
print("Target Variable 'y' Distribution After Transformation:")
print(df["y"].value_counts())

# Step 5: Save the Transformed Data to a New CSV File
output_path = "transformed_bank_data.csv"
df.to_csv(output_path, index=False)
print(f"Data transformation completed and saved to '{output_path}'")
