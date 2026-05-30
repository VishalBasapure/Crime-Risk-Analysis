
import pandas as pd
import numpy as np
import re
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────
 
df = pd.read_csv('south_crime_raw.csv')
 
print("=== BEFORE CLEANING ===")
print(f"  Rows    : {len(df)}")
print(f"  Columns : {list(df.columns)}")
print()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 2 — RENAME COLUMNS to match crime_map_data
# ─────────────────────────────────────────────────────────────
 
df = df.rename(columns={
    '5Police Station' : 'district',
    'Year'            : 'year',
    'Type'            : 'crime_type',
    'Date'            : 'date',
    'Time'            : 'time',
    'Place'           : 'place',
    'Latitude'        : 'latitude',
    'Longitude'       : 'longitude',
})
 
print("Columns after rename:", list(df.columns))
print()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 3 — COORDINATE PARSER
# ─────────────────────────────────────────────────────────────
 
BENGALURU_LAT = (12.7, 13.2)
BENGALURU_LON = (77.3, 78.0)
 
 
def parse_dms(s):
    """
    Convert standard DMS to decimal degrees.
    Handles: 12°55'3.65"N  and  77°34'6.67"E
 
    Math:
        decimal = degrees + minutes/60 + seconds/3600
        12°55'3.65"N = 12 + 55/60 + 3.65/3600 = 12.9177
    """
    m = re.match(
        r"(\d+)[°]\s*(\d+)['\u2019]\s*([0-9.]+)[\"s]?\s*([NSEWnsew])?",
        s
    )
    if not m:
        return None
    deg, mins, secs, direction = m.groups()
    try:
        decimal = float(deg) + float(mins) / 60 + float(secs) / 3600
    except ValueError:
        return None
    if direction and direction.upper() in ('S', 'W'):
        decimal = -decimal
    return round(decimal, 7)
 
 
def recover_shifted(f, lo, hi):
    """
    Recover a coord where the decimal point shifted.
    Example: 12921652  → 12.921652  (÷ 1,000,000)
             77560741  → 77.560741  (÷ 1,000,000)
              7756466  → 77.56466   (÷   100,000)
 
    Try common divisors until the result is in the expected range.
    """
    for divisor in [1_000_000, 100_000, 10_000_000]:
        candidate = round(f / divisor, 7)
        if lo < candidate < hi:
            return candidate
    return None
 
 
def parse_coord(raw_value, lo, hi):
    """
    Master parser — tries every format, returns float or None.
 
    lo, hi = valid range for this axis
             lat: (12.7, 13.2)   lon: (77.3, 78.0)
    """
    if pd.isna(raw_value):
        return None
 
    s = str(raw_value).strip()
 
    if re.match(r'^[\d.]+°$', s):
        s = s.rstrip('°').strip()

    if '°' in s:
        result = parse_dms(s)
        if result is not None and lo < result < hi:
            return result
        # DMS present but malformed — drop it
        return None
 
    # ── Type E: trailing dot or internal space ────────────────
    # "12.940815."  →  "12.940815"
    # "12. 928365"  →  "12.928365"
    s = s.rstrip('.').replace('. ', '.').replace(' ', '')
 
    # ── Type A / D: plain float ───────────────────────────────
    try:
        f = float(s)
 
        # Type A — already valid decimal in Bengaluru
        if lo < f < hi:
            return round(f, 7)
 
        # Type D — shifted decimal (value is thousands× too large)
        if f > hi * 100:
            return recover_shifted(f, lo, hi)
 
        # Valid float but wrong city (e.g. Chennai at 13.1, 80.2)
        return None
 
    except (ValueError, TypeError):
        # Type F — completely non-numeric, drop
        return None
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 4 — APPLY PARSER
# ─────────────────────────────────────────────────────────────
 
print("Parsing coordinates...")
 
# Track counts before
before = len(df)
 
df['latitude']  = df['latitude'].apply(
    lambda v: parse_coord(v, *BENGALURU_LAT)
)
df['longitude'] = df['longitude'].apply(
    lambda v: parse_coord(v, *BENGALURU_LON)
)
 
# Drop rows where EITHER coordinate could not be recovered
df = df.dropna(subset=['latitude', 'longitude'])
 
print(f"  Rows before : {before}")
print(f"  Rows after  : {len(df)}")
print(f"  Dropped     : {before - len(df)} rows (unrecoverable coordinates)")
print()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 5 — CLEAN crime_type
#  WHY: crime_map_data stores 'murder' (lowercase).
#       south_crime_raw stores 'Murder' (title case).
#       LabelEncoder and filters treat them as different values.
#       This one line makes them identical.
# ─────────────────────────────────────────────────────────────
 
df['crime_type'] = df['crime_type'].astype(str).str.lower().str.strip()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 6 — CLEAN REMAINING COLUMNS
# ─────────────────────────────────────────────────────────────
 
# district — lowercase to match crime_map_data police station names
df['district'] = df['district'].astype(str).str.lower().str.strip()
 
# year — force to integer (was float 2016.0 in raw file)
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
 
# date — standardise to YYYY-MM-DD to match crime_map_data
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
 
# time — strip whitespace
df['time'] = df['time'].astype(str).str.strip()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 7 — KEEP ONLY COLUMNS THAT MATCH crime_map_data
#  crime_map_data has: district, crime_type, latitude,
#                      longitude, date, time
#  We add 'year' as a bonus column (useful for filtering by year)
# ─────────────────────────────────────────────────────────────
 
df_final = df[[
    'district', 'crime_type', 'latitude', 'longitude', 'date', 'time', 'year'
]].copy()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 8 — VALIDATION
# ─────────────────────────────────────────────────────────────
 
print("=== VALIDATION REPORT ===")
print(f"  Total rows          : {len(df_final)}")
print(f"  Null latitudes      : {df_final['latitude'].isna().sum()}")
print(f"  Null longitudes     : {df_final['longitude'].isna().sum()}")
print(f"  Null crime_type     : {df_final['crime_type'].isna().sum()}")
print(f"  Lat range           : {df_final['latitude'].min():.5f} – "
      f"{df_final['latitude'].max():.5f}")
print(f"  Lon range           : {df_final['longitude'].min():.5f} – "
      f"{df_final['longitude'].max():.5f}")
print()
print("  Crime type counts:")
print(df_final['crime_type'].value_counts().to_string())
print()
 
lat_ok = df_final['latitude'].between(*BENGALURU_LAT).all()
lon_ok = df_final['longitude'].between(*BENGALURU_LON).all()
print(f"  All lats in Bengaluru range : {'YES' if lat_ok else 'NO — CHECK!'}")
print(f"  All lons in Bengaluru range : {'YES' if lon_ok else 'NO — CHECK!'}")
print()
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 9 — SAVE
# ─────────────────────────────────────────────────────────────
 
df_final.to_csv('south_crime_raw_CLEAN.csv', index=False)
print(f"Saved  →  south_crime_raw_CLEAN.csv  ({len(df_final)} rows)")
print()
print("Next step: run CODE 2 (merge script) using this file.")