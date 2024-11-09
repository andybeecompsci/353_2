# this file is used to debug t3, keep getting singular column error, tryna figure out if the transformed data set
# is wrong 

import pandas as pd

# load the original dataset
file_path = "bank-additional-full.csv"
df_original = pd.read_csv(file_path, sep=";")

# check the distribution of the 'y' column
print("Original Target Variable 'y' Distribution:")
print(df_original["y"].value_counts())
