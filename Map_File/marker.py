
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster


CRIME_FILTER = 'all'

KDE_BANDWIDTH = 0.015
MAX_MARKERS = 3500
MAP_TILE ='CartoDB positron'
attr = 'Esri'


# ─────────────────────────────────────────────────────────────
#  STEP 1  —  LOAD DATA
# ─────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv('bengaluru_master.csv')
print(f"  {len(df):,} rows loaded")
print(f"  Crime types:\n{df['crime_type'].value_counts().to_string()}")
print()


# ─────────────────────────────────────────────────────────────
#  STEP 2  —  FILTER BY CRIME TYPE
# ─────────────────────────────────────────────────────────────

if CRIME_FILTER == 'all':
    filtered  = df.copy()
    map_title = 'All Crimes'
else:
    filtered  = df[df['crime_type'] == CRIME_FILTER].copy()
    map_title = CRIME_FILTER.title()

if len(filtered) == 0:
    print(f"No rows found for '{CRIME_FILTER}'.")
    print(f"Available: {sorted(df['crime_type'].unique())}")
    raise SystemExit

print(f"Filtered to '{CRIME_FILTER}': {len(filtered):,} incidents")


# ─────────────────────────────────────────────────────────────
#  STEP 3  —  EXTRACT COORDINATES
# ─────────────────────────────────────────────────────────────

lats = filtered['latitude'].values
lons = filtered['longitude'].values


# ─────────────────────────────────────────────────────────────
#  STEP 4  —  RUN KDE  (Kernel Density Estimation)
# ─────────────────────────────────────────────────────────────

print("Running KDE...")

coords = np.vstack([lats, lons])          # shape (2, N)
kde    = gaussian_kde(coords, bw_method=KDE_BANDWIDTH)

# Evaluation grid — covers Bengaluru with a small margin
lat_min, lat_max = lats.min() - 0.01, lats.max() + 0.01
lon_min, lon_max = lons.min() - 0.01, lons.max() + 0.01

lat_grid, lon_grid = np.mgrid[
    lat_min:lat_max:170j,
    lon_min:lon_max:170j
]

grid_coords = np.vstack([lat_grid.ravel(), lon_grid.ravel()])
density     = kde(grid_coords).reshape(lat_grid.shape)

print(f"  KDE complete  |  max density: {density.max():.4f}")

# ─────────────────────────────────────────────────────────────
#  STEP 5  —  CREATE FOLIUM MAP
# ─────────────────────────────────────────────────────────────

print("Building map...")

m = folium.Map(
    location=[12.9716, 77.5946],   # Bengaluru city centre
    zoom_start=12,
    tiles=MAP_TILE,
    attr=attr
)


# ─────────────────────────────────────────────────────────────
#  STEP 8  —  ADD CRIME MARKERS  (capped at MAX_MARKERS)
# ─────────────────────────────────────────────────────────────

PRIORITY_TYPES = ['murder', 'rape', 'attempted murder', 'robbery']

priority_rows = filtered[filtered['crime_type'].isin(PRIORITY_TYPES)]
other_rows    = filtered[~filtered['crime_type'].isin(PRIORITY_TYPES)]

priority_cap    = min(len(priority_rows), int(MAX_MARKERS * 0.6))
other_cap       = MAX_MARKERS - priority_cap

priority_sample = (priority_rows.sample(n=priority_cap, random_state=42)
                   if len(priority_rows) > priority_cap else priority_rows)
other_sample    = other_rows.sample(
                      n=min(other_cap, len(other_rows)), random_state=42)

markers_df = pd.concat([priority_sample, other_sample])

print(f"  Markers: {len(markers_df)} total  "
      f"(priority {len(priority_sample)}, others {len(other_sample)})")

# Colour map — one distinct colour per crime type
CRIME_COLORS = {
    'murder'           : 'red',
    'rape'             : 'darkred',
    'attempted murder' : 'orange',
    'robbery'          : 'darkblue',
    'assault'          : 'purple',
    'kidnap'           : 'lime',
    'chain snatching'  : 'cyan',
    '2 wheeler theft'  : 'cadetblue',
    '4 wheeler theft'  : 'lightblue',
    'ordinary theft'   : 'gray',
}

marker_cluster = MarkerCluster().add_to(m)

for _, row in markers_df.iterrows():
    crime = row['crime_type']
    year  = int(row['year']) if pd.notna(row['year']) else 'unknown'
    color = CRIME_COLORS.get(crime, 'white')

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=1,
        popup=f"{crime} ({year})"
    ).add_to(marker_cluster)

# ─────────────────────────────────────────────────────────────
#  STEP 9  —  TITLE OVERLAY
# ─────────────────────────────────────────────────────────────

title_html = f"""
<div style="
    position:fixed; top:14px; left:50%; transform:translateX(-50%);
    z-index:1000; background:rgba(20,20,30,0.85); color:white;
    padding:8px 20px; border-radius:8px;
    font-family:Arial,sans-serif; font-size:14px; font-weight:bold;
    border:1px solid rgba(255,255,255,0.15);">
  Bengaluru Crime Hotspot — {map_title}
  &nbsp;|&nbsp;
  <span style="font-weight:normal; font-size:12px;">
    {len(filtered):,} incidents &nbsp;·&nbsp; {len(markers_df)} markers shown
  </span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))


# ─────────────────────────────────────────────────────────────
#  STEP 10  —  LEGEND
# ─────────────────────────────────────────────────────────────

legend_html = """
<div style="
    position:fixed; bottom:30px; right:14px; z-index:1000;
    background:rgba(20,20,30,0.85); color:white;
    padding:12px 16px; border-radius:8px;
    font-family:Arial,sans-serif; font-size:12px;
    border:1px solid rgba(255,255,255,0.15); min-width:150px;">
  <b style="display:block;margin-top:10px;padding-top:8px;
     border-top:1px solid rgba(255,255,255,0.2);">Markers</b>
  <div style="margin-top:6px;">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:red;margin-right:6px;"></span>Murder<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:orange;margin-right:6px;margin-top:3px;"></span>Att. murder<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:darkblue;margin-right:6px;margin-top:3px;"></span>Robbery<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:purple;margin-right:6px;margin-top:3px;"></span>Assault<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:lightblue;margin-right:6px;margin-top:3px;"></span>Vehicle theft<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:lime;margin-right:6px;margin-top:3px;"></span>Kidnaap<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:cyan;margin-right:6px;margin-top:3px;"></span>chain snatching<br>
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
          background:grey;margin-right:6px;margin-top:3px;"></span>Other theft
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))


# ─────────────────────────────────────────────────────────────
#  STEP 11  —  SAVE
# ─────────────────────────────────────────────────────────────

output_file = f"bengaluru_marker_{CRIME_FILTER.replace(' ', '_')}.html"
m.save(output_file)

print()
print("=" * 48)
print(f"  Saved  :  {output_file}")
print(f"  Rows   :  {len(filtered):,}")
print(f"  Markers:  {len(markers_df)}")
print("=" * 48)
print("Open the .html file in Chrome / Firefox.")
print()
print("To map a single crime type, change CRIME_FILTER")
print("at the top of this file and run again.")