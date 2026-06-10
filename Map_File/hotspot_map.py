
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster


CRIME_FILTER = 'all'

# KDE precision.  Lower = sharper hotspots.  Higher = broader blobs.
KDE_BANDWIDTH = 0.015

# Heatmap visual settings
HEATMAP_RADIUS  = 25   
HEATMAP_BLUR    = 15    
HEATMAP_OPACITY = 0.3   


MAP_TILE ='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
attr = 'Esri'


# ─────────────────────────────────────────────────────────────
#  STEP 1  —  LOAD DATA
# ─────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv('../data/processed/bengaluru_master.csv')
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
#  STEP 5  —  BUILD HEAT DATA
# ─────────────────────────────────────────────────────────────

threshold = density.max() * 0.01
heat_data = []

for i in range(lat_grid.shape[0]):
    for j in range(lat_grid.shape[1]):
        if density[i, j] > threshold:
            heat_data.append([
                lat_grid[i, j],
                lon_grid[i, j],
                float(density[i, j])
            ])

print(f"  {len(heat_data)} heat points above threshold")


# ─────────────────────────────────────────────────────────────
#  STEP 6  —  CREATE FOLIUM MAP
# ─────────────────────────────────────────────────────────────

print("Building map...")

m = folium.Map(
    location=[12.9716, 77.5946],   # Bengaluru city centre
    zoom_start=12,
    tiles=MAP_TILE,
    attr=attr
)


# ─────────────────────────────────────────────────────────────
#  STEP 7  —  ADD HEATMAP LAYER
# ─────────────────────────────────────────────────────────────

HeatMap(
    heat_data,
    min_opacity = HEATMAP_OPACITY,
    max_zoom    = 18,
    radius      = HEATMAP_RADIUS,
    blur        = HEATMAP_BLUR,
    gradient = {
    0.1: 'blue',
    0.3: 'cyan',
    0.5: 'lime',
    0.7: 'yellow',
    0.9: 'orange',
    1.0: 'red'
}
).add_to(m)


# ─────────────────────────────────────────────────────────────
#  STEP 8  —  TITLE OVERLAY
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
    {len(filtered):,} incidents &nbsp;·&nbsp;
  </span>
</div>
<div style="
    position:fixed; bottom:30px; right:14px; z-index:1000;
    background:rgba(20,20,30,0.85); color:white;
    padding:12px 16px; border-radius:8px;
    font-family:Arial,sans-serif; font-size:12px;
    border:1px solid rgba(255,255,255,0.15); min-width:150px;">
  <b>Heatmap intensity</b>
  <div style="margin-top:6px;">
    <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
          background:red;margin-right:6px;"></span>Hotspot<br>
    <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
          background:orange;margin-right:6px;margin-top:4px;"></span>High<br>
    <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
          background:lime;margin-right:6px;margin-top:4px;"></span>Medium<br>
    <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
          background:blue;margin-right:6px;margin-top:4px;"></span>Low
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))


# ─────────────────────────────────────────────────────────────
#  STEP 10 —  SAVE
# ─────────────────────────────────────────────────────────────

output_file = f"../outputs-maps/bengaluru_hotspot_{CRIME_FILTER.replace(' ', '_')}.html"
m.save(output_file)

print()
print("=" * 48)
print(f"  Saved  :  {output_file}")
print(f"  Rows   :  {len(filtered):,}")
print(f"  Heat pts: {len(heat_data)}")
print("=" * 48)
print("Open the .html file in Chrome / Firefox.")
print()
print("To map a single crime type, change CRIME_FILTER")
print("at the top of this file and run again.")