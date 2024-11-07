import pandas as pd

# Load the original dataset
file_path = "bank-additional-full.csv"
df_original = pd.read_csv(file_path, sep=";")

# Check the distribution of the 'y' column
print("Original Target Variable 'y' Distribution:")
print(df_original["y"].value_counts())
