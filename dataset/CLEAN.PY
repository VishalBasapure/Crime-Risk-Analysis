import pandas as pd

# -------------------------------
# 1. LOAD DATA
# -------------------------------
crime = pd.read_csv("crime_standardized.csv")
ml = pd.read_csv("ml_standardized.csv")
map1 = pd.read_csv("map1.csv")

# -------------------------------
# 2. CLEAN COLUMN NAMES
# -------------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

crime = clean_columns(crime)
ml = clean_columns(ml)
map1 = clean_columns(map1)

# -------------------------------
# 3. RENAME COLUMNS
# -------------------------------
if "crime_count" in ml.columns:
    ml = ml.rename(columns={"crime_count": "count"})

# -------------------------------
# 4. NORMALIZE TEXT
# -------------------------------
def normalize(df):
    df["district"] = df["district"].astype(str).str.lower().str.strip()
    df["crime_type"] = df["crime_type"].astype(str).str.lower().str.strip()
    return df

crime = normalize(crime)
ml = normalize(ml)
map1 = normalize(map1)

# -------------------------------
# 5. DROP NaNs
# -------------------------------
crime = crime.dropna(subset=["district", "year", "crime_type", "count"])
ml = ml.dropna(subset=["district", "year", "crime_type", "police_deployed"])
map1 = map1.dropna(subset=["district", "year", "crime_type", "police_deployed"])

# -------------------------------
# 6. TYPE FIX
# -------------------------------
crime["year"] = crime["year"].astype(int)
ml["year"] = ml["year"].astype(int)
map1["year"] = map1["year"].astype(int)

# -------------------------------
# 7. CRIME TYPE SIMPLIFICATION (CRITICAL FIX)
# -------------------------------

import re

def normalize_crime_type(x):
    x = str(x).lower()

    # -------------------------------
    # REMOVE LEGAL + NOISE TEXT
    # -------------------------------
    x = re.sub(r"sec.*", "", x)  # remove sections
    x = re.sub(r"\(.*?\)", "", x)  # remove brackets
    x = re.sub(r"[^a-z\s]", "", x)  # remove symbols/numbers
    x = x.strip()

    # -------------------------------
    # STANDARDIZATION RULES
    # -------------------------------
    if "kidnapping" in x or "abduction" in x:
        return "kidnapping"

    elif "murder" in x:
        return "murder"

    elif "rape" in x or "sexual" in x:
        return "rape"

    elif "theft" in x and "vehicle" in x:
        return "vehicle theft"

    elif "theft" in x:
        return "theft"

    elif "robbery" in x or "dacoity" in x:
        return "robbery"

    elif "cyber" in x or "technology" in x:
        return "cyber crime"

    elif "dowry" in x:
        return "dowry"

    elif "assault" in x or "hurt" in x or "violence" in x:
        return "assault"

    elif "child" in x or "pocso" in x:
        return "child crime"

    elif "women" in x:
        return "women crime"

    elif "fraud" in x or "cheating" in x or "forgery" in x:
        return "fraud"

    else:
        return x  # KEEP ORIGINAL (IMPORTANT — no over-shrinking)
crime["crime_type"] = crime["crime_type"].apply(normalize_crime_type)
ml["crime_type"] = ml["crime_type"].apply(normalize_crime_type)
map1["crime_type"] = map1["crime_type"].apply(normalize_crime_type)
# -------------------------------
# 8. REMOVE DUPLICATES (AGGREGATE)
# -------------------------------
crime = crime.groupby(
    ["district", "year", "crime_type"],
    as_index=False
)["count"].sum()
ml = ml.groupby(
    ["district", "year", "crime_type"],
    as_index=False
).agg({
    "count": "sum",
    "police_deployed": "mean"
})

map1 = map1.groupby(
    ["district", "year", "crime_type"],
    as_index=False
)["police_deployed"].mean()

# -------------------------------
# 9. COMBINE POLICE DATA
# -------------------------------
police_data = pd.concat([
    ml[["district", "crime_type", "police_deployed"]],
    map1[["district", "crime_type", "police_deployed"]]
], ignore_index=True)

# -------------------------------
# 10. AGGREGATE POLICE DATA
# -------------------------------
police_agg = police_data.groupby(
    ["district", "crime_type"],
    as_index=False
)["police_deployed"].mean()

# -------------------------------
# 11. MERGE (RELAXED)
# -------------------------------
final = crime.merge(
    police_agg,
    on=["district", "crime_type"],
    how="left"
)

# -------------------------------
# 12. SMART FILL (MULTI LEVEL)
# -------------------------------
# Crime type level
final["police_deployed"] = final["police_deployed"].fillna(
    final.groupby("crime_type")["police_deployed"].transform("mean")
)

# District level
final["police_deployed"] = final["police_deployed"].fillna(
    final.groupby("district")["police_deployed"].transform("mean")
)

# Global level
final["police_deployed"] = final["police_deployed"].fillna(
    final["police_deployed"].mean()
)

# -------------------------------
# 13. FINAL VALIDATION
# -------------------------------
print("\nFINAL SHAPE:", final.shape)
print("\nNaN CHECK:\n", final.isnull().sum())
print("\nPOLICE STATS:\n", final["police_deployed"].describe())
print("\nDISTRICTS:", final["district"].nunique())
print("\nCRIME TYPES:", final["crime_type"].unique())

# -------------------------------
# 14. SAVE FINAL DATASET
# -------------------------------
final.to_csv("MASTER_DATASET_FINAL_READY.csv", index=False)

print("\n✅ DONE: MASTER DATASET READY FOR ML 🚀")