import pandas as pd
import numpy as np 

"""df=pd.read_csv("final_karnataka_crime.csv")
df["district"]=df["district"].str.lower().str.strip()
df["crime_type"]=df["crime_type"].str.lower().str.strip()
df=df.drop(["state"], axis=1)
df.to_csv("cleaned_karnataka_crime.csv", index=False)
print(df.head())"""

df1=pd.read_csv("../data/raw/cleaned_karnataka_crime.csv")
df2=pd.read_csv("../data/raw/crime_standardized.csv")

df1=df1.groupby(['district', 'year', 'crime_type'], as_index=False)['count'].sum()
df2 = df2.groupby(['district', 'year', 'crime_type'], as_index=False)['count'].sum()

df_final=pd.concat([df1, df2], ignore_index=True)
df_final.to_csv("../data/processed/final_karnataka_crime.csv", index=False)

print(df_final.count().sum())
# Check duplicates
print("Duplicates:", df_final.duplicated().sum())

# Check districts
print("Total Districts:", df_final['district'].nunique())
print(df_final['district'].unique())

# Check years
print("Years:", sorted(df_final['year'].unique()))

# Missing values
print(df_final.isnull().sum())