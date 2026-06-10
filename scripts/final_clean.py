
import pandas as pd
df = pd.read_csv('../data/processed/final_karnataka_crime.csv')

print("=== BEFORE CLEANING ===")
print(f"  Rows        : {len(df)}")
print(f"  Crime types : {df['crime_type'].nunique()}")
print()

df = df[~df['crime_type'].str.contains('total', case=False)].copy()

print(f"After removing aggregates : {len(df)} rows")
print(f"Crime types remaining     : {df['crime_type'].nunique()}")
print()

VIOLENT = [
    'murder (sec.302 ipc)',
    'attempt to commit murder (sec.307 ipc)',
    'grievous hurt (sec.325, 32,326a, 326b,329,331, 333, 335 ipc)',
    'simple hurt (sec.323 r/w ipc,324,332,353,327,328,330 ipc)',
    'acid attack (sec. 326a ipc)',
    'attempt to acid attack (sec. 326b ipc)',
]


WOMEN = [
    'rape (sec. 376 ipc)',
    'attempt to commit rape (sec. 376/511 ipc)',
    'murder with rape/gang rape',
    'dowry deaths (sec. 304b ipc)',
    'dowry prohibition act, 1961',
    'cruelty by husband or his relatives (sec. 498 a ipc)',
    'assault on women with intent to outrage her modesty (sec. 354 ipc)',
    'kidnapping & abduction of women',
    'abetment to suicide of women (sec. 305/306 ipc)',
    'immoral traffic (prevention) act 1956 (women victims cases only)',
    'insult to the modesty of women (sec. 509 ipc)',
    'indecent representation of women (prohibition) act, 1986',
    'cyber crimes/information technology act (women centric crimes only)',
    'human trafficking (sec. 370 & 370a ipc)',
]

CHILDREN = [
    'kidnapping and abduction of children',
    'protection of children from sexual offences act (pocso) r/w sec.376,354, 509 ipc)',
    'protection of children from sexual violence act (girl child victims only)',
    'child labour (prohibition & regulation) act',
    'juveniles justice (care and protection of children) act',
    'abetment of suicide of child (sec.305 ipc)',
    'human trafficking (secs. 370 & 370a ipc) (children only)',
    'exposure and abandonment (sec.317 ipc)',
    'foeticide (sec. 315 & 316 ipc)',
    'infanticide (sec.315 ipc)',
    'prohibition of child marriage act',
    'miscarriage (sec. 313 & 314 ipc)',
]


CYBER = [
    'cyber crimes/information technology act',
]


OTHER = [
    'other ipc crimes',
    'other sll crimes',
]


category_map = {}
for ct in VIOLENT:  category_map[ct] = 'violent_crime'
for ct in WOMEN:    category_map[ct] = 'crime_against_women'
for ct in CHILDREN: category_map[ct] = 'crime_against_children'
for ct in CYBER:    category_map[ct] = 'cyber_crime'
for ct in OTHER:    category_map[ct] = 'other_crime'

df['crime_category'] = df['crime_type'].map(category_map)
unmapped = df[df['crime_category'].isna()]['crime_type'].unique()
if len(unmapped) > 0:
    print(f"WARNING — unmapped crime types ({len(unmapped)}):")
    for u in unmapped:
        print(f"  {u}")
else:
    print("All crime types mapped successfully — 0 unmapped")
print()
#reorder columns for better readability
df = df[['district', 'year', 'crime_type', 'crime_category', 'count']]

df.to_csv('../data/processed/karnataka_categorised.csv', index=False)
print(f"\nSaved → ../data/processed/karnataka_categorised.csv ({len(df)} rows, {df.shape[1]} columns)")
print("Columns:", list(df.columns))