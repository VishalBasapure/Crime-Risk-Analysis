import pandas as pd
#feature selection and pivoting
df = pd.read_csv("data/processed/karnataka_categorised.csv")
df = df.groupby(['district', 'year', 'crime_category'], as_index=False)['count'].sum()
df_pivot = df.pivot_table(
    index=['district', 'year'],
    columns='crime_category',
    values='count',
    aggfunc='sum',
    fill_value=0
).reset_index()

print(df_pivot.head())
#missing value check
for col in ['violent_crime', 'crime_against_women', 'crime_against_children', 'cyber_crime', 'other_crime']:
    if col not in df_pivot.columns:
        df_pivot[col] = 0
        
#feature engineering - total crime count
df_pivot['total_crime'] = df_pivot[
    ['violent_crime', 'crime_against_women', 'crime_against_children', 'cyber_crime', 'other_crime']
].sum(axis=1)

df_pivot['variety'] = (
    df_pivot[['violent_crime', 'crime_against_women', 'crime_against_children', 'cyber_crime', 'other_crime']] > 0
).sum(axis=1)

df_pivot['wc_ratio'] = (
    df_pivot['crime_against_women'] + df_pivot['crime_against_children']
) / df_pivot['total_crime']

df_pivot['wc_ratio'] = df_pivot['wc_ratio'].fillna(0)

df_pivot = df_pivot.sort_values(['district', 'year'])

df_pivot['trend'] = df_pivot.groupby('district')['total_crime'].pct_change()
df_pivot['trend'] = df_pivot['trend'].fillna(0)
print("\nFeature Engineered Data:")
print(df_pivot.head())

#label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_pivot['district'] = le.fit_transform(df_pivot['district'])
print("\nLabel Encoded Districts:")
print(df_pivot['district'].unique())
print(df_pivot.head())

#feature selection
X = df_pivot[['district', 'year', 'trend', 'variety', 'wc_ratio']]
y = df_pivot[['violent_crime', 'crime_against_women', 'crime_against_children', 'cyber_crime', 'other_crime']]

#data spliting for training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

df_pivot.to_csv('data/processed/karnataka_dataset.csv', index=False)



