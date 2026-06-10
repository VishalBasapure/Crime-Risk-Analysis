# ============================================================
#  CODE 2 — CLEAN crime_map_data + MERGE BOTH FILES
#  Input  : crime_map_data.csv
#           south_crime_raw_CLEAN.csv  (output of Code 1)
#  Output : bengaluru_master.csv
#
#  What this does:
#    1. Cleans crime_map_data coordinates (same garbage patterns)
#    2. Aligns columns between both files
#    3. Merges them
#    4. Deduplicates properly (by location + crime, not by row)
#    5. Validates the final dataset
# ============================================================

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
#  STEP 1 — LOAD BOTH FILES
# ─────────────────────────────────────────────────────────────

df_map = pd.read_csv('../data/output_data/crime_map_data.csv')
df_raw = pd.read_csv('south_crime_raw_CLEAN.csv')

print("=== LOADED FILES ===")
print(f"  crime_map_data rows      : {len(df_map)}")
print(f"  south_crime_raw_CLEAN    : {len(df_raw)}")
print()
print(f"  crime_map_data columns   : {list(df_map.columns)}")
print(f"  south_crime_raw columns  : {list(df_raw.columns)}")
print()


# ─────────────────────────────────────────────────────────────
#  STEP 2 — CLEAN crime_map_data COORDINATES
#
#  WHY: crime_map_data has the SAME coordinate problems:
#    - 148 rows with shifted decimals (12879498 instead of 12.8795)
#    - Some rows with wrong-city coordinates (Chennai, Kerala)
#
#  We use the same recover_shifted logic from Code 1.
# ─────────────────────────────────────────────────────────────

BENGALURU_LAT = (12.7, 13.2)
BENGALURU_LON = (77.3, 78.0)


def recover_shifted(f, lo, hi):
    """
    Recover a coord where the decimal point shifted.
    Tries dividing by common powers of 10 until result
    falls inside the expected Bengaluru range.
    """
    for divisor in [1_000_000, 100_000, 10_000_000]:
        candidate = round(f / divisor, 7)
        if lo < candidate < hi:
            return candidate
    return None


def clean_coord(raw_value, lo, hi):
    """
    Clean a single coordinate value from crime_map_data.
    Returns float if valid/recoverable, None if not.
    """
    if pd.isna(raw_value):
        return None
    try:
        f = float(raw_value)

        # Already valid
        if lo < f < hi:
            return round(f, 7)

        # Shifted decimal — try to recover
        if f > hi * 100:
            return recover_shifted(f, lo, hi)

        # Valid number but wrong city (Chennai, Kerala etc.)
        # Cannot recover — must drop
        return None

    except (ValueError, TypeError):
        return None


print("Cleaning crime_map_data coordinates...")
before_map = len(df_map)

df_map['latitude']  = df_map['latitude'].apply(
    lambda v: clean_coord(v, *BENGALURU_LAT)
)
df_map['longitude'] = df_map['longitude'].apply(
    lambda v: clean_coord(v, *BENGALURU_LON)
)

df_map = df_map.dropna(subset=['latitude', 'longitude'])

print(f"  Before : {before_map}")
print(f"  After  : {len(df_map)}")
print(f"  Dropped: {before_map - len(df_map)} garbage coord rows")
print()


# ─────────────────────────────────────────────────────────────
#  STEP 3 — STANDARDISE crime_map_data COLUMNS
#  WHY: crime_map_data columns need to match south_crime_raw_CLEAN
#       so concat lines them up correctly.
#       crime_map_data has: district, crime_type, latitude,
#                           longitude, date, time
#       We add a year column extracted from date.
# ─────────────────────────────────────────────────────────────

# Lowercase crime_type — should already be but enforce it
df_map['crime_type'] = df_map['crime_type'].astype(str).str.lower().str.strip()

# Lowercase district
df_map['district'] = df_map['district'].astype(str).str.lower().str.strip()

# Extract year from date column
df_map['year'] = pd.to_datetime(
    df_map['date'], errors='coerce'
).dt.year.astype('Int64')

# Standardise date format to YYYY-MM-DD
df_map['date'] = pd.to_datetime(
    df_map['date'], errors='coerce'
).dt.strftime('%Y-%m-%d')

# Keep only aligned columns
df_map = df_map[[
    'district', 'crime_type', 'latitude', 'longitude', 'date', 'time', 'year'
]].copy()

print("crime_map_data after standardising:")
print(f"  Shape   : {df_map.shape}")
print(f"  Columns : {list(df_map.columns)}")
print()


# ─────────────────────────────────────────────────────────────
#  STEP 4 — MERGE
#  WHY concat not merge:
#    pd.merge joins on a KEY (like SQL JOIN).
#    pd.concat stacks rows vertically (like SQL UNION).
#    We want UNION — stack all rows from both files,
#    then deduplicate — so concat is correct here.
# ─────────────────────────────────────────────────────────────

print("Merging files...")

df_merged = pd.concat([df_map, df_raw], ignore_index=True)

print(f"  After concat (before dedup) : {len(df_merged)} rows")


# ─────────────────────────────────────────────────────────────
#  STEP 5 — DEDUPLICATE PROPERLY
#
#  WHY not just drop_duplicates():
#    drop_duplicates() with no arguments compares EVERY column.
#    df_map has NaN in the 'year' column for some rows.
#    df_raw has the same crime with year filled in.
#    They look "different" row-by-row so both survive.
#    Result: same crime counted twice → inflated hotspots.
#
#  WHY round coordinates:
#    12.930433 and 12.9304330 are the same GPS point but
#    Python sees them as different floats due to precision.
#    Rounding to 4 decimal places = ~11 metre precision.
#    Two readings within 11m of each other = same incident.
#
#  WHY subset=['crime_type', 'lat_r', 'lon_r']:
#    Same location could have different crimes on different days.
#    We only collapse rows where BOTH location AND crime match.
# ─────────────────────────────────────────────────────────────

df_merged['lat_r'] = df_merged['latitude'].round(4)
df_merged['lon_r'] = df_merged['longitude'].round(4)

before_dedup = len(df_merged)

df_merged = df_merged.drop_duplicates(
    subset=['crime_type', 'lat_r', 'lon_r'],
    keep='first'     # keep the first occurrence (crime_map_data rows come first)
)

# Drop the helper rounding columns — not needed in final file
df_merged = df_merged.drop(columns=['lat_r', 'lon_r'])

print(f"  After dedup                 : {len(df_merged)} rows")
print(f"  Duplicates removed          : {before_dedup - len(df_merged)}")
print()


# ─────────────────────────────────────────────────────────────
#  STEP 6 — FINAL VALIDATION
# ─────────────────────────────────────────────────────────────

print("=== FINAL VALIDATION REPORT ===")
print(f"  Total rows          : {len(df_merged)}")
print(f"  Null latitudes      : {df_merged['latitude'].isna().sum()}")
print(f"  Null longitudes     : {df_merged['longitude'].isna().sum()}")
print(f"  Null crime_type     : {df_merged['crime_type'].isna().sum()}")
print(f"  Lat range           : {df_merged['latitude'].min():.5f} – "
      f"{df_merged['latitude'].max():.5f}")
print(f"  Lon range           : {df_merged['longitude'].min():.5f} – "
      f"{df_merged['longitude'].max():.5f}")
print(f"  Year range          : "
      f"{df_merged['year'].min()} – {df_merged['year'].max()}")
print()
print("  Crime type distribution:")
print(df_merged['crime_type'].value_counts().to_string())
print()
print("  Year distribution:")
print(df_merged['year'].value_counts().sort_index().to_string())
print()

# Bounding box check
lat_ok = df_merged['latitude'].between(*BENGALURU_LAT).all()
lon_ok = df_merged['longitude'].between(*BENGALURU_LON).all()
print(f"  All lats in Bengaluru bbox  : {'YES' if lat_ok else 'NO — CHECK!'}")
print(f"  All lons in Bengaluru bbox  : {'YES' if lon_ok else 'NO — CHECK!'}")
print()


# ─────────────────────────────────────────────────────────────
#  STEP 7 — SAVE
# ─────────────────────────────────────────────────────────────

df_merged.to_csv('../data/processed/bengaluru_master.csv', index=False)

print(f"Saved  →  ../data/processed/bengaluru_master.csv  ({len(df_merged)} rows)")
print()
print("Feed ../data/processed/bengaluru_master.csv into your KDE hotspot script.")
print("Change CRIME_FILTER in the hotspot script to map any crime type.")