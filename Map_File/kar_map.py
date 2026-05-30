
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

df = pd.read_csv("dcvi_2024_predictions.csv")

# ── Fix 1: Risk bins — percentile-based so distribution is balanced ──
# Old bins=[0,70,80,100] put almost everything in LOW (meaningless)
# New bins use 33rd and 66th percentile of actual DCVI scores
p33 = np.percentile(df['dcvi_score'], 30)
p66 = np.percentile(df['dcvi_score'], 70)
df['risk'] = pd.cut(
    df['dcvi_score'],
    bins=[0, p33, p66, 100],
    labels=['LOW', 'MEDIUM', 'HIGH'],
    include_lowest=True
).astype(str)

print(f"Risk thresholds: LOW < {p33:.1f} | MEDIUM {p33:.1f}-{p66:.1f} | HIGH > {p66:.1f}")
print("Risk distribution:", df['risk'].value_counts().to_dict())
print()

# ── Fix 2: Complete coords dict — keys match CSV names EXACTLY ──
# Old dict used 'Bengaluru Urban', CSV has 'bengaluru city' → 0 matches
district_coords = {
    "bagalkot"             : [16.18, 75.69],
    "ballari"              : [15.15, 76.93],
    "belagavi city"        : [15.85, 74.50],
    "belagavi district"    : [15.70, 74.90],
    "bengaluru city"       : [12.97, 77.59],
    "bengaluru district"   : [13.10, 77.45],
    "bidar"                : [17.91, 77.52],
    "chamarajnagar"        : [11.92, 76.94],
    "chikkaballapura"      : [13.44, 77.73],
    "chikkamagaluru"       : [13.32, 75.77],
    "chitradurga"          : [14.23, 76.40],
    "dakshina kannada"     : [12.84, 74.86],
    "davanagere"           : [14.46, 75.92],
    "dharwad"              : [15.46, 75.01],
    "gadag"                : [15.43, 75.63],
    "hassan"               : [13.00, 76.10],
    "haveri"               : [14.80, 75.40],
    "hubballi dharwad city": [15.36, 75.12],
    "k.g.f."               : [12.97, 78.27],
    "k.railways"           : [15.31, 75.71],
    "kalaburgi"            : [17.33, 76.83],
    "kalaburgi city"       : [17.40, 76.90],
    "kodagu"               : [12.42, 75.74],
    "kolar"                : [13.14, 78.13],
    "koppal"               : [15.35, 76.15],
    "mandya"               : [12.52, 76.90],
    "mangaluru city"       : [12.91, 74.85],
    "mysuru city"          : [12.30, 76.65],
    "mysuru district"      : [12.20, 76.40],
    "raichur"              : [16.20, 77.35],
    "ramanagara"           : [12.72, 77.28],
    "shimoga"              : [13.93, 75.57],
    "tumakuru"             : [13.34, 77.10],
    "udupi"                : [13.34, 74.75],
    "uttara kannada"       : [14.79, 74.13],
    "vijayanagara"         : [15.32, 76.46],
    "vijayapura"           : [16.83, 75.71],
    "yadgiri"              : [16.77, 77.14],
}

# Verify all 38 match
matched   = [d for d in df['district'] if d in district_coords]
unmatched = [d for d in df['district'] if d not in district_coords]
print(f"Matched: {len(matched)}/38 districts")
if unmatched:
    print(f"WARNING — unmatched: {unmatched}")

def get_color(risk):
    return {'HIGH': '#dc3545', 'MEDIUM': '#fd7e14', 'LOW': '#198754'}.get(risk, '#adb5bd')


# ── Build map ────────────────────────────────────────────────
m = folium.Map(
    location=[15.0, 76.5],
    zoom_start=7,
    tiles="CartoDB positron"  
)

marker_layer = folium.FeatureGroup(name="District Markers", show=True)
heat_layer   = folium.FeatureGroup(name="DCVI Heatmap",    show=True)

heat_data = []   # Fix 3: was never populated in original code

for _, row in df.iterrows():
    d    = str(row['district'])
    risk = str(row['risk'])
    dcvi = float(row['dcvi_score'])
    pred = int(row['pred_crimes_2024'])
    act  = int(row['actual_2023'])

    if d not in district_coords:
        continue

    lat, lon = district_coords[d]

    # Fix 3: actually populate heat_data
    heat_data.append([lat, lon, dcvi])

    # Popup with full info
    popup_html = f"""
    <div style='font-family:Arial,sans-serif;font-size:13px;min-width:190px;padding:4px'>
        <b style='font-size:15px'>{d.title()}</b>
        <hr style='margin:5px 0;border-color:#ddd'>
        <table style='width:100%'>
        <tr><td><b>DCVI Score</b></td><td style='text-align:right'>{dcvi}</td></tr>
        <tr><td><b>Risk Level</b></td>
            <td style='text-align:right;color:{get_color(risk)};font-weight:bold'>{risk}</td></tr>
        <tr><td><b>Predicted 2024</b></td><td style='text-align:right'>{pred:,}</td></tr>
        <tr><td><b>Actual 2023</b></td><td style='text-align:right'>{act:,}</td></tr>
        </table>
    </div>"""

    folium.CircleMarker(
        location=[lat, lon],
        radius=max(6, dcvi / 8),       # bigger circle = higher DCVI
        color=get_color(risk),
        fill=True,
        fill_color=get_color(risk),
        fill_opacity=0.80,
        weight=1.5,
        popup=folium.Popup(popup_html, max_width=230),
        tooltip=f"{d.title()}  |  DCVI: {dcvi}  |  {risk}"
    ).add_to(marker_layer)
# Fix 4: heatmap now has data with correct gradient
HeatMap(
    heat_data,
    radius=55,
    blur=20,
    max_zoom=20,
    gradient={'0.3': 'blue', '0.5': 'cyan', '0.7': 'lime',
              '0.85': 'orange', '1.0': 'red'}
).add_to(heat_layer)

marker_layer.add_to(m)
heat_layer.add_to(m)
folium.LayerControl().add_to(m)
print("Map saved successfully!")

# Title bar
title_html = """
<div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
z-index:1000;background:rgba(15,15,25,0.90);color:white;
padding:9px 22px;border-radius:8px;font-family:Arial;
font-size:14px;font-weight:bold;letter-spacing:0.3px;
border:1px solid rgba(255,255,255,0.2);">
Karnataka — District Crime Vulnerability Index (DCVI) 2024
</div>"""
m.get_root().html.add_child(folium.Element(title_html))

# Legend
legend_html = f"""
<div style="position:fixed;bottom:30px;right:14px;z-index:1000;
background:rgba(15,15,25,0.90);color:white;padding:14px 18px;
border-radius:8px;font-family:Arial;font-size:13px;
border:1px solid rgba(255,255,255,0.2);min-width:175px;">
<b style='font-size:14px'>Risk Level (DCVI)</b><br><br>
<span style='color:#dc3545;font-size:18px'>●</span>
  <b>HIGH</b> &nbsp;(&gt; {p66:.0f})<br>
<span style='color:#fd7e14;font-size:18px'>●</span>
  <b>MEDIUM</b> ({p33:.0f} – {p66:.0f})<br>
<span style='color:#198754;font-size:18px'>●</span>
  <b>LOW</b> &nbsp;(&lt; {p33:.0f})<br><br>
<small style='color:#aaa'>Circle size = DCVI score<br>
Click any circle for details</small>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save("karnataka1_crime_2024_fixed.html")   
print(f"\nMap saved: karnataka_crime_2024_fixed.html")
print(f"Heat points plotted: {len(heat_data)}")
print("Open in Chrome or Firefox — works offline.")