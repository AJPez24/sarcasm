# DATA EXPLORATION

import csv
import pandas as pd

# file with raw data, labels - sarcastic = 1 (only 2009-2012)
raw = "sarc_09_12.csv" 


df = pd.read_csv(
    raw,
    nrows=1000,
    low_memory=False,
    on_bad_lines='skip',
    quotechar='"'
)

#print(df.head())

columns = list(df.columns.values)
print("Columns :", columns)