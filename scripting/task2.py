import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# CLEAAANINGGGG TIMEEEEEEE

# load the original dataset
file_path = "bank-additional-full.csv"
df = pd.read_csv(file_path, sep=";")

# separate features (x) and target variable (y)
y = df["y"].apply(lambda x: 1 if x == "yes" else 0)  # convert target to binary
X = df.drop(columns=["y"])

# identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# create transformers for numerical and categorical data
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  # handle missing values
        ("scaler", StandardScaler()),  # standardize numerical data
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="missing"),
        ),  # fill missing with 'missing'
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),  # one-hot encode categorical data
    ]
)

# combine transformers into a preprocessor using columntransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# apply transformations to the features
X_transformed = preprocessor.fit_transform(X)
feature_names = (
    preprocessor.get_feature_names_out()
)  # get feature names after transformation

# convert transformed features to dataframe
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
X_transformed_df["y"] = y  # add target variable back

# save the new dataset with transformed features
X_transformed_df.to_csv("transformed_bank_data.csv", index=False)
print("task 2 complete. transformed data saved to 'transformed_bank_data.csv'.")
