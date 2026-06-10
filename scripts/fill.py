
import pandas as pd
import numpy as np

np.random.seed(2024)   # reproducible, year-based seed

df = pd.read_csv('../data/processed/karnataka_dataset.csv')

print("=== BEFORE FILL ===")
print(f"Rows where crime_against_women == 0 : {(df['crime_against_women']==0).sum()}")
print()

# ── Real NCRB Karnataka growth rates ──────────────────────
RATE_2020_TO_2021 = 1.153   # +15.3% national (2020 was COVID dip year)
RATE_2021_TO_2022 = 1.231   # +23.1% Karnataka state actual (14468→17813)

# WHY different for 2022:
# We know Karnataka's actual 2022 total from NCRB = 17,813
# Karnataka 2021 total = 14,468
# So real growth = 17813/14468 = 1.231 (+23.1%) NOT +4%
# The +4% is national — Karnataka was worse than national avg

NOISE_STD = 0.06     # ±6% std. dev — reflects real district variation
NOISE_CAP = 0.10     # ±10% hard cap


# ── District-wise fill loop ────────────────────────────────
df_filled = df.copy()

for district_id in df['district'].unique():
    mask = df_filled['district'] == district_id

    # Anchor: 2021 is real data
    val_2021 = df_filled.loc[
        mask & (df_filled['year'] == 2021), 'crime_against_women'
    ].values[0]

    if val_2021 == 0:
        print(f"  WARNING: district {district_id} has 2021=0, skipping")
        continue

    # ── 2020: back-extrapolate ────────────────────────────
    # 2021 = 2020 × 1.153  →  2020 = 2021 / 1.153
    base_2020 = val_2021 / RATE_2020_TO_2021
    noise_2020 = np.clip(np.random.normal(1.0, NOISE_STD), 1-NOISE_CAP, 1+NOISE_CAP)
    val_2020   = max(round(base_2020 * noise_2020), 1)  # never produce 0

    # ── 2022: forward-extrapolate ──────────────────────────
    # 2022 = 2021 × 1.231
    base_2022 = val_2021 * RATE_2021_TO_2022
    noise_2022 = np.clip(np.random.normal(1.0, NOISE_STD), 1-NOISE_CAP, 1+NOISE_CAP)
    val_2022   = max(round(base_2022 * noise_2022), 1)

    # ── Write back ─────────────────────────────────────────
    df_filled.loc[
        mask & (df_filled['year'] == 2020), 'crime_against_women'
    ] = val_2020
    df_filled.loc[
        mask & (df_filled['year'] == 2022), 'crime_against_women'
    ] = val_2022


# ── Recalculate total_crime ────────────────────────────────
df_filled['total_crime'] = (
    df_filled['crime_against_women'] +
    df_filled['crime_against_children'] +
    df_filled['cyber_crime'] +
    df_filled['other_crime'] +
    df_filled['violent_crime']
)

# ── Recalculate wc_ratio ──────────────────────────────────
df_filled['wc_ratio'] = (
    (df_filled['crime_against_women'] + df_filled['crime_against_children'])
    / df_filled['total_crime'].replace(0, 1)
)

# ── Recalculate trend (YoY growth per district) ──────────
df_filled = df_filled.sort_values(['district', 'year'])
df_filled['lag1'] = df_filled.groupby('district')['total_crime'].shift(1)
df_filled['trend'] = (
    (df_filled['total_crime'] - df_filled['lag1'])
    / df_filled['lag1'].replace(0, 1)
).fillna(0).round(6)
df_filled = df_filled.drop(columns=['lag1'])


# ── Validation ────────────────────────────────────────────
print("=== AFTER FILL — VALIDATION ===")
print(f"Zeros remaining       : {(df_filled['crime_against_women']==0).sum()}")
print(f"Negative values       : {(df_filled['crime_against_women']<0).sum()}")
print()

print("State totals by year (should match NCRB roughly):")
totals = df_filled.groupby('year')['crime_against_women'].sum()
print(totals)
print()
print("NCRB Karnataka reference: 2021=14,468  2022=17,813")
print(f"Our 2021 total: {totals[2021]:,}  (should be ~14,468)")
print(f"Our 2022 total: {totals[2022]:,}  (should be ~17,813)")
print()

print("Real growth rates in filled data:")
state_2020 = totals[2020]
state_2021 = totals[2021]
state_2022 = totals[2022]
state_2023 = totals[2023]
print(f"  2020→2021 : {(state_2021/state_2020-1)*100:+.1f}%  (NCRB says +15.3%)")
print(f"  2021→2022 : {(state_2022/state_2021-1)*100:+.1f}%  (NCRB says +23.1% for Karnataka)")
print(f"  2022→2023 : {(state_2023/state_2022-1)*100:+.1f}%  (2023 is synthetic ×2 of 2021)")
print()

print("Sample district timelines:")
for dist_id, dist_name in [(0,'bagalkot'), (4,'bengaluru city'), (10,'chitradurga'), (22,'kodagu')]:
    row = df_filled[df_filled['district']==dist_id][[
        'year','crime_against_women','total_crime','wc_ratio'
    ]]
    print(f"\n  {dist_name}:")
    print(row.to_string(index=False))


# ── Save ──────────────────────────────────────────────────
df_filled.to_csv('../data/raw/karnataka_filled_v2.csv', index=False)
print(f"\nSaved → ../data/raw/karnataka_filled_v2.csv  ({len(df_filled)} rows)")
print("Columns:", list(df_filled.columns))
print()
print("=== KNOWN LIMITATIONS — document these in your report ===")
print("1. 2020 and 2022 women crime values are imputed.")
print("   Method: district 2021 value × NCRB Karnataka growth rates")
print("   (−13.3% for 2020, +23.1% for 2022) with ±6% Gaussian noise.")
print("2. 2023 women crime values are mechanically doubled from 2021")
print("   in the source dataset. This is a known data quality issue.")
print("3. Actual district-level breakdown for 2020/2022 from NCRB")
print("   was not available. State-level rates were applied uniformly.")