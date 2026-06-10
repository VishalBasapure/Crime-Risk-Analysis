import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
try:
    df = pd.read_csv("data/processed/karnataka_categorised.csv")
except FileNotFoundError:
    print("Error: data/processed/karnataka_categorised.csv not found.")
    exit(1)
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

le = LabelEncoder()
df_pivot['district'] = le.fit_transform(df_pivot['district'])
print("\nLabel Encoded Districts:")
print(df_pivot['district'].unique())
print(df_pivot.head())

#feature selection
X = df_pivot[['district', 'year', 'trend', 'variety', 'wc_ratio']]
y = df_pivot[['violent_crime', 'crime_against_women', 'crime_against_children', 'cyber_crime', 'other_crime']]

#data spliting for training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

df_pivot.to_csv('data/processed/karnataka_dataset.csv', index=False)




#Step 6 + 7 —DCVI Score + Train Random Forest + Predict 2024


# ─── LOAD ──────────────────────────────────────────────────
try:
    df  = pd.read_csv('data/raw/karnataka_filled_v2.csv')
    df2 = pd.read_csv('data/processed/karnataka_categorised_final.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
dist_map = dict(enumerate(sorted(df2['district'].unique())))

# ─── STEP 5: FEATURE ENGINEERING ──────────────────────────
# Add 4 new columns that give the model historical context

df = df.sort_values(['district', 'year'])

# lag_1_count: last year's total crime for this district
# WHY: history is the strongest predictor of future crime
df['lag_1_count'] = df.groupby('district')['total_crime'].shift(1).fillna(0)
df['lag_2_count'] = df.groupby('district')['total_crime'].shift(2).fillna(0)
df['lag_3_count'] = df.groupby('district')['total_crime'].shift(3).fillna(0)
df['rolling_mean_3'] = (
    df.groupby('district')['total_crime']
      .rolling(3).mean()
      .reset_index(0, drop=True)
      .fillna(0)
)
# lag_1_women / lag_1_children: last year's category breakdown
df['lag_1_women']    = df.groupby('district')['crime_against_women'].shift(1).fillna(0)
df['lag_1_children'] = df.groupby('district')['crime_against_children'].shift(1).fillna(0)

# yoy_women: year-over-year % change in women crime
# WHY: tells model if district is getting worse or better
df['yoy_women'] = (
    df.groupby('district')['crime_against_women']
      .pct_change().fillna(0).clip(-1, 2).round(4)
)

# is_odd_year: 1 if 2021/2023, 0 if 2020/2022
# WHY: NCRB alternates which crime categories are published each year
# This flag tells the model which pattern to expect
df['is_odd_year'] = (df['year'] % 2).astype(int)

# ─── STEP 6: DCVI — build target column ───────────────────
# DCVI = District Crime Vulnerability Index (0-100)
# Calculated from 4 weighted sub-scores
#
# We fit scalers on TRAIN data only (2020-2022)
# WHY: if you fit on all data, test info leaks into training
#      and the model cheats → looks good but predicts badly

TRAIN_YEARS  = [2020, 2021, 2022]
EXCLUDE_DIST = [19]  # k.railways — not a real district (18 crimes total)

train_mask = df['year'].isin(TRAIN_YEARS) & ~df['district'].isin(EXCLUDE_DIST)
train = df[train_mask].copy()
test  = df[df['year'] == 2023].copy()

# Fit scalers on TRAIN only
sc_vol = MinMaxScaler(feature_range=(0, 100)).fit(train[['total_crime']])
sc_wc  = MinMaxScaler(feature_range=(0, 100)).fit(train[['wc_ratio']])
sc_tr  = MinMaxScaler(feature_range=(0, 100)).fit(train[['yoy_women']])
sc_var = MinMaxScaler(feature_range=(0, 100)).fit(train[['variety']])

def compute_dcvi(data, sc_vol, sc_wc, sc_tr, sc_var):
    d = data.copy()
    d['s_volume']  = sc_vol.transform(d[['total_crime']]).round(2)
    d['s_wc']      = sc_wc.transform(d[['wc_ratio']]).round(2)
    d['s_trend']   = sc_tr.transform(d[['yoy_women']]).round(2)
    d['s_variety'] = sc_var.transform(d[['variety']]).round(2)
    d['dcvi_score'] = (
        d['s_volume']  * 0.35 +   # 35% weight — volume of crime
        d['s_wc']      * 0.35 +   # 35% weight — women+children ratio
        d['s_trend']   * 0.20 +   # 20% weight — rising/falling trend
        d['s_variety'] * 0.10     # 10% weight — crime variety
    ).round(1)
    return d

train = compute_dcvi(train, sc_vol, sc_wc, sc_tr, sc_var)

# ─── STEP 7: TRAIN RANDOM FOREST ──────────────────────────
# Target: total_crime (predict the actual number)
# WHY not DCVI directly: DCVI is built FROM the features,
# predicting it directly causes circular dependency and data leakage.
# We predict total_crime, then compute DCVI from the prediction.

FEATURES = [
    'district', 'year', 'is_odd_year',
    'crime_against_children', 'crime_against_women',
    'cyber_crime', 'other_crime', 'violent_crime',
    'variety', 'wc_ratio',
    'lag_1_count', 'lag_2_count', 'lag_3_count',
    'lag_1_women', 'lag_1_children',
    'yoy_women', 'rolling_mean_3'
]
TARGET = 'total_crime'

X_train = train[FEATURES]
y_train = train[TARGET]

rf = RandomForestRegressor(
    n_estimators=1500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluate on 2023 (test year, excluding k.railways)
test_eval = test[~test['district'].isin(EXCLUDE_DIST)]
mae = mean_absolute_error(test_eval[TARGET], rf.predict(test_eval[FEATURES]))
r2  = r2_score(test_eval[TARGET],            rf.predict(test_eval[FEATURES]))
accuracy = r2 * 100
print(f"Model Performance on 2023:")
print(f"  MAE : {mae:.0f} crimes  (average error per district)")
print(f"  R2  : {r2:.3f}  (model explains {r2 * 100:.1f}% of variance)")
print(f"  Accuracy : {accuracy:.2f}%")
print()

# Save metrics for the dashboard
with open('data/output_data/dcvi_metrics.json', 'w', encoding='utf-8') as f:
    json.dump({'r2': float(r2), 'mae': float(mae)}, f)

# Feature importance
print("Feature importance:")
for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.3f}")
print()

# ─── PREDICT 2024 ──────────────────────────────────────────
pred_2024 = test.copy()
pred_2024['pred_total'] = rf.predict(pred_2024[FEATURES]).clip(1).round(0)

# Compute DCVI from predicted totals
pred_col = pred_2024[['pred_total']].rename(columns={'pred_total': 'total_crime'})
pred_2024['s_volume']   = sc_vol.transform(pred_col).round(2)
pred_2024['s_wc']       = sc_wc.transform(pred_2024[['wc_ratio']]).round(2)
pred_2024['s_trend']    = sc_tr.transform(pred_2024[['yoy_women']]).round(2)
pred_2024['s_variety']  = sc_var.transform(pred_2024[['variety']]).round(2)
pred_2024['dcvi_2024']  = (
    pred_2024['s_volume']  * 0.35 +
    pred_2024['s_wc']      * 0.35 +
    pred_2024['s_trend']   * 0.20 +
    pred_2024['s_variety'] * 0.10
).round(1)

pred_2024['risk_2024'] = pd.cut(
    pred_2024['dcvi_2024'],
    bins=[0, 70, 80, 100],
    labels=['LOW', 'MEDIUM', 'HIGH'],
    include_lowest=True
)
pred_2024['district_name'] = pred_2024['district'].map(dist_map)

output = pred_2024[['district_name', 'dcvi_2024', 'risk_2024', 'pred_total', 'total_crime']].copy()
output.columns = ['district', 'dcvi_score', 'risk', 'pred_crimes_2024', 'actual_2023']
output = output.sort_values('dcvi_score', ascending=False)

print("=== 2024 DISTRICT PREDICTIONS ===")
print(output.to_string(index=False))
print()
print("Risk distribution:", pred_2024['risk_2024'].value_counts().to_dict())

# ─── SAVE ALL OUTPUTS ──────────────────────────────────────
output.to_csv('data/output_data/dcvi_2024_predictions.csv', index=False)
joblib.dump(rf,    'models/dcvi_model.pkl')
joblib.dump({'vol': sc_vol, 'wc': sc_wc, 'tr': sc_tr, 'var': sc_var}, 'models/dcvi_scalers.pkl')
joblib.dump(dist_map, 'models/district_map.pkl')
df.to_csv('data/processed/karnataka_model_ready.csv', index=False)

print("\nFiles saved:")
print("  dcvi_model.pkl           — trained Random Forest")
print("  dcvi_scalers.pkl         — MinMaxScaler objects")
print("  district_map.pkl         — district number → name mapping")
print("  dcvi_2024_predictions.csv— 38 districts ranked by DCVI")
print("  karnataka_model_ready.csv— full dataset with all features")









#test data set 
# ─── TEST DATASET ─────────────────────────────
print()
print("=== TEST DATASET ===")

test_cases = pd.DataFrame([
    {
        'district': 4,
        'year': 2024,
        'is_odd_year': 0,
        'crime_against_children': 100,
        'crime_against_women': 200,
        'cyber_crime': 150,
        'other_crime': 800,
        'violent_crime': 300,
        'variety': 5,
        'wc_ratio': (200+100)/(200+100+150+800+300),
        'lag_1_count': 1500,
        'lag_2_count': 1400,
        'lag_3_count': 1300,
        'lag_1_women': 180,
        'lag_1_children': 90,
        'yoy_women': 0.05,
        'rolling_mean_3': 1400,
        'expected_total': 1550
    },
    {
        'district': 4,
        'year': 2024,
        'is_odd_year': 0,
        'crime_against_children': 500,
        'crime_against_women': 900,
        'cyber_crime': 700,
        'other_crime': 2500,
        'violent_crime': 1200,
        'variety': 5,
        'wc_ratio': (900+500)/(900+500+700+2500+1200),
        'lag_1_count': 5500,
        'lag_2_count': 5200,
        'lag_3_count': 5000,
        'lag_1_women': 850,
        'lag_1_children': 450,
        'yoy_women': 0.20,
        'rolling_mean_3': 5200,
        'expected_total': 5800
    }
])

preds = rf.predict(test_cases[FEATURES])

#PERFORMANCE 
sample_r2 = r2_score(test_cases['expected_total'], preds)
sample_mae = mean_absolute_error(test_cases['expected_total'], preds)

print(f"Custom Test R2       : {sample_r2:.3f}")
print(f"Custom Test Accuracy : {sample_r2*100:.2f}%")
print(f"Custom Test MAE      : {sample_mae:.2f}")

#DCVI CALCULATION 
test_cases['pred_total'] = preds

test_cases['s_volume'] = sc_vol.transform(
    test_cases[['pred_total']].rename(columns={'pred_total': 'total_crime'})
)

# Normalize
test_cases['s_volume'] = (test_cases['pred_total'] / test_cases['pred_total'].max()) * 100
test_cases['s_wc'] = test_cases['wc_ratio'] * 100
test_cases['s_trend'] = test_cases['yoy_women'] * 100
test_cases['s_variety'] = (test_cases['variety'] / 5) * 100

test_cases['dcvi_score'] = (
    test_cases['s_volume'] * 0.35 +
    test_cases['s_wc'] * 0.35 +
    test_cases['s_trend'] * 0.20 +
    test_cases['s_variety'] * 0.10
).clip(0,100).round(1)

for i, row in test_cases.iterrows():
    print()
    print(f"--- Test Case {i+1} ---")
    print(f"Expected Crime      : {row['expected_total']}")
    print(f"Predicted Crime     : {row['pred_total']:.0f}")
    print(f"Predicted DCVI      : {row['dcvi_score']}")