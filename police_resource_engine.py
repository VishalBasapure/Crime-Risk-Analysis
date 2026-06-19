import streamlit as st
import pandas as pd


# ── KGIS district code → canonical (old) district name ────────────────────────
# These are the 30 administrative police districts as coded in the KML files.
KGIS_TO_DISTRICT = {
    '29555': 'bagalkot',        '29556': 'bangalore rural',
    '29557': 'bangalore urban', '29558': 'belgaum',
    '29559': 'bellary',         '29560': 'bidar',
    '29561': 'bijapur',         '29562': 'chamarajanagara',
    '29563': 'chikkaballapura', '29564': 'chikkamagaluru',
    '29565': 'chitradurga',     '29566': 'dakshina kannada',
    '29567': 'davanagere',      '29568': 'dharwad',
    '29569': 'gadag',           '29570': 'kalaburagi',
    '29571': 'hassan',          '29572': 'haveri',
    '29573': 'kodagu',          '29574': 'kolar',
    '29575': 'koppal',          '29576': 'mandya',
    '29577': 'mysuru',          '29578': 'raichur',
    '29579': 'ramanagara',      '29580': 'shivamogga',
    '29581': 'tumakuru',        '29582': 'udupi',
    '29583': 'uttara kannada',  '29584': 'yadgiri',
}

# ── Aliases: old KGIS name → list of equivalent / renamed forms used elsewhere ─
# Karnataka renamed several districts (2014) and some prediction datasets split
# a single police-district into "City" + "District/Rural" pairs that don't exist
# as separate codes in the KML data. We normalize both sides to a common root
# so e.g. "belagavi city" and "belagavi district" both inherit the Belgaum count.
DISTRICT_ALIASES = {
    'belgaum':           ['belgaum', 'belagavi'],
    'bellary':           ['bellary', 'ballari'],
    'bijapur':           ['bijapur', 'vijayapura'],
    'bangalore rural':   ['bangalore rural', 'bengaluru rural'],
    'bangalore urban':   ['bangalore urban', 'bengaluru urban', 'bengaluru city',
                           'bengaluru', 'bangalore'],
    'kalaburagi':        ['kalaburagi', 'gulbarga', 'kalaburgi'],
    'chamarajanagara':   ['chamarajanagara', 'chamarajanagar', 'chamrajnagar',
                           'chamarajnagar'],
    'mysuru':            ['mysuru', 'mysore'],
    'shivamogga':        ['shivamogga', 'shimoga'],
    'tumakuru':          ['tumakuru', 'tumkur'],
    'yadgiri':           ['yadgiri', 'yadgir'],
    'uttara kannada':    ['uttara kannada', 'uttar kannada'],
    'dharwad':           ['dharwad', 'hubballi dharwad'],
    'kolar':             ['kolar', 'kgf', 'k.g.f', 'k.g.f.'],
    'dakshina kannada':  ['dakshina kannada', 'mangaluru', 'mangalore'],
}

# Special policing jurisdictions that are NOT geographic districts (e.g. railway
# police, which patrols rail corridors rather than a district). These genuinely
# have zero stations in the geographic KGIS dataset — that's correct, not a bug —
# so we flag them to render an honest "N/A — special jurisdiction" label instead
# of a misleading zero.
SPECIAL_JURISDICTIONS = {'k.railways', 'k railways', 'krailways', 'railways'}


def _normalize_name(name):
    """Strip common suffixes/variants so split district names map to one root."""
    n = str(name).lower().strip()
    n = n.replace('district', '').replace('rural', '').replace('city', '')
    n = n.replace('.', '').replace('  ', ' ').strip()
    return n


def is_special_jurisdiction(district_name):
    return _normalize_name(district_name) in SPECIAL_JURISDICTIONS


def _build_root_lookup():
    """Build {normalized_alias_or_kgis_name: kgis_root_name} for every known
    spelling variant, so any incoming prediction-district string can be resolved
    back to the correct KGIS police-data root."""
    lookup = {}
    for kgis_root in KGIS_TO_DISTRICT.values():
        lookup[_normalize_name(kgis_root)] = kgis_root
    for kgis_root, aliases in DISTRICT_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_name(alias)] = kgis_root
    return lookup


ROOT_LOOKUP = _build_root_lookup()


def resolve_to_kgis_root(district_name):
    """Map any prediction-district spelling to its KGIS police-data root name.
    Falls back to the normalized name itself if no alias is found (so unmatched
    districts are still visible/debuggable rather than silently zeroed)."""
    norm = _normalize_name(district_name)
    return ROOT_LOOKUP.get(norm, norm)


@st.cache_data
def load_police_data():
    """Load police stations & outposts, count per district, compute coverage metrics."""
    try:
        ps = pd.read_csv("police_data/kml_extracted_2.csv")
        po = pd.read_csv("police_data/kml_extracted_1.csv")
    except FileNotFoundError:
        # fallback paths
        ps = pd.read_csv("kml_extracted_2.csv")
        po = pd.read_csv("kml_extracted_1.csv")

    ps = ps.dropna(subset=['KGISCode']).copy()
    po = po.dropna(subset=['KGISCode']).copy()

    ps['dist_code'] = ps['KGISCode'].astype(str).str[:5]
    po['dist_code'] = po['KGISCode'].astype(str).str[:5]

    ps['district'] = ps['dist_code'].map(KGIS_TO_DISTRICT)
    po['district'] = po['dist_code'].map(KGIS_TO_DISTRICT)

    ps_counts = ps.groupby('district').size().reset_index(name='police_stations')
    po_counts = po.groupby('district').size().reset_index(name='police_outposts')

    police_df = ps_counts.merge(po_counts, on='district', how='outer').fillna(0)
    police_df['police_stations']  = police_df['police_stations'].astype(int)
    police_df['police_outposts']  = police_df['police_outposts'].astype(int)

    # Keep station & outpost geo coords for mapping
    ps_geo = ps[['district','POL_STAName','Latitude','Longitude']].rename(
        columns={'POL_STAName': 'name'})
    ps_geo['type'] = 'Station'
    po_geo = po[['district','POL_OPSTName','Latitude','Longitude']].rename(
        columns={'POL_OPSTName': 'name'})
    po_geo['type'] = 'Outpost'
    police_geo = pd.concat([ps_geo, po_geo], ignore_index=True)

    return police_df, police_geo


def compute_coverage(pred_df, police_df):
    """Merge crime predictions with police data and compute Coverage Index & Gap Score.

    Districts that were split into City/Rural/District variants in the
    predictions file (but not in the KGIS police data) are resolved back to
    their shared root via resolve_to_kgis_root(), so e.g. 'bengaluru city' and
    'bengaluru district' both correctly inherit the Bangalore Urban count
    instead of one of them showing zero.
    """
    df = pred_df.copy()
    df['kgis_root'] = df['district'].apply(resolve_to_kgis_root)
    df['is_special'] = df['district'].apply(is_special_jurisdiction)

    police = police_df.copy()
    police['kgis_root'] = police['district'].apply(
        lambda d: _normalize_name(d))   # police_df['district'] is already a KGIS root

    df = df.merge(police[['kgis_root','police_stations','police_outposts']],
                  on='kgis_root', how='left')
    df['police_stations']  = df['police_stations'].fillna(0).astype(int)
    df['police_outposts']  = df['police_outposts'].fillna(0).astype(int)

    # When a single root district was split (e.g. city + rural variants both
    # mapped to 'bangalore urban'), each split row gets the FULL station/outpost
    # count of the shared root above. Halve the count proportional to how many
    # prediction-rows share that root, so total infrastructure isn't double-counted
    # across the dashboard's summary stats.
    split_counts = df.groupby('kgis_root')['district'].transform('count')
    df['police_stations'] = (df['police_stations'] / split_counts).round().astype(int)
    df['police_outposts'] = (df['police_outposts'] / split_counts).round().astype(int)

    # Coverage Index: weighted police presence relative to crime load
    # Formula: ((stations * 2) + outposts) / predicted_crimes * 1000
    # Normalized 0-100 using min-max
    df['raw_coverage'] = ((df['police_stations'] * 2 + df['police_outposts'])
                           / df['pred_crimes_2024'].clip(lower=1) * 1000)
    cmin, cmax = df['raw_coverage'].min(), df['raw_coverage'].max()
    df['coverage_index'] = ((df['raw_coverage'] - cmin) / (cmax - cmin) * 100).round(1)

    # Gap Score: high risk + low coverage = highest priority
    risk_score = df['risk'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1})
    coverage_inv = 100 - df['coverage_index']           # inverted: low coverage = high gap
    dcvi_norm = df['dcvi_score'] / 100

    df['gap_score'] = (risk_score * 30 + coverage_inv * 0.5 + dcvi_norm * 20).round(1)
    gmin, gmax = df['gap_score'].min(), df['gap_score'].max()
    df['gap_score'] = ((df['gap_score'] - gmin) / (gmax - gmin) * 100).round(1)

    return df


def build_recommendations(row):
    """Rule-based recommendation engine for a single district row."""
    recs = []

    # ── Thresholds (absolute crime counts, 2023 actuals) ──────────────
    high_women    = row.get('crime_against_women', 0) > 300
    high_children = row.get('crime_against_children', 0) > 150
    high_cyber    = row.get('cyber_crime', 0) > 20
    high_violent  = row.get('violent_crime', 0) > 50
    risk          = row.get('risk', 'LOW')
    cov_idx       = row.get('coverage_index', 50)
    low_coverage  = cov_idx < 35

    if risk == 'HIGH' and low_coverage:
        recs.append(("🚨 Infrastructure Priority",
                     "Establish new police stations in underserved zones. "
                     "Deploy mobile patrol units immediately. "
                     "Request district-level force augmentation."))

    if risk in ('HIGH', 'MEDIUM') and low_coverage:
        recs.append(("📍 Patrol Enhancement",
                     "Increase beat patrol frequency by 40%. "
                     "Deploy night patrol vehicles on high-crime corridors. "
                     "Install CCTV surveillance at key junctions."))

    if high_women:
        recs.append(("👩 Women Safety",
                     "Deploy dedicated women's patrol units (Vanitha teams). "
                     "Install emergency helpline boards near schools, colleges & markets. "
                     "Conduct awareness programs on 1091 Women Helpline & Sakhi centres."))

    if high_children:
        recs.append(("🧒 Child Protection",
                     "Activate school safety committees and gate-monitoring programs. "
                     "Run community awareness on POCSO Act and ChildLine 1098. "
                     "Coordinate with DCPU for child protection follow-up."))

    if high_cyber:
        recs.append(("💻 Cyber Crime",
                     "Set up Cyber Crime awareness workshops in colleges & offices. "
                     "Increase staffing at district Cyber Crime cell. "
                     "Run digital fraud prevention campaigns via local media."))

    if high_violent:
        recs.append(("⚔️ Violent Crime",
                     "Strengthen quick-response units in identified hotspot areas. "
                     "Increase preventive arrests under goondas / externment orders. "
                     "Improve inter-PS communication for real-time incident response."))

    if not recs:
        recs.append(("✅ Maintain Current Coverage",
                     "District shows relatively low crime vulnerability. "
                     "Continue routine patrol schedules and community policing. "
                     "Focus on preventive outreach and beat verification."))

    return recs