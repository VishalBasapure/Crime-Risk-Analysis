# ============================================================
#  app.py — Karnataka Crime Risk Prediction Dashboard
#  Integrates Model A (DCVI) + Model B (Bengaluru hotspot)
#
#  Install:
#    pip install streamlit pandas numpy scikit-learn folium
#               streamlit-folium plotly joblib
#
#  Run:
#    streamlit run app.py
#
#  Files needed in same folder:
#    dcvi_2024_predictions.csv
#    karnataka_model_ready.csv
#    dcvi_model.pkl
#    dcvi_scalers.pkl
#    district_map.pkl
#    karnataka_crime_2024_fixed.html   (run kar_map_fixed.py first)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import warnings, os, json
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Karnataka Crime Risk Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; font-size: 18px; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* ── Global text size boost ── */
p, li, span, div, label { font-size: 18px !important; }
h1 { font-size: 2.8rem !important; }
h2 { font-size: 2.2rem !important; }
h3 { font-size: 1.9rem !important; }

/* ── Selectbox / radio labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label { font-size: 18px !important; font-weight: 600; }
[data-testid="stSelectbox"] div[data-baseweb="select"] { font-size: 18px !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 18px 22px !important;
}
[data-testid="metric-container"] label {
    font-size: 16px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6c757d !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 16px !important;
}

/* ── Risk badge ── */
.badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 20px;
    font-size: 32px !important;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-HIGH   { background:#fde8e8;color:#b91c1c;border:1.5px solid #f87171; }
.badge-MEDIUM { background:#fef3cd;color:#92400e;border:1.5px solid #fbbf24; }
.badge-LOW    { background:#d1fae5;color:#065f46;border:1.5px solid #34d399; }

/* ── Section headings ── */
.sec-head {
    font-size: 25px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6c757d;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1.5px solid #e9ecef;
}

/* ── Info box ── */
.info-box {
    background: #f0f4ff;
    border-left: 4px solid #4f46e5;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 26px !important;
    color: #3730a3;
    margin-bottom: 14px;
}

/* ── Dataframe text ── */
[data-testid="stDataFrame"] { font-size: 32px !important; }
[data-testid="stDataFrame"] table { font-size: 32px !important; }
[data-testid="stDataFrame"] table th,
[data-testid="stDataFrame"] table td,
[data-testid="stDataFrame"] table span {
    font-size: 32px !important;
    padding: 12px 14px !important;
}
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td {
    font-size: 32px !important;
    padding: 12px 14px !important;
}
[data-testid="stDataFrame"] .row_heading,
[data-testid="stDataFrame"] .col_heading {
    font-size: 32px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1e1e2e; }
[data-testid="stSidebar"] * { color: #cdd6f4 !important; font-size: 30px !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-size: 30px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    pred     = pd.read_csv("dcvi_2024_predictions.csv")
    model_df = pd.read_csv("karnataka_model_ready.csv")
    dist_names   = sorted(pred['district'].unique())
    dist_num_map = dict(enumerate(dist_names))

    pred['change_pct']       = ((pred['pred_crimes_2024'] - pred['actual_2023'])
                                 / pred['actual_2023'] * 100).round(1)
    pred['pred_crimes_2025'] = (pred['pred_crimes_2024']
                                 * (1 + pred['change_pct'] / 100)).round(0)

    p33 = np.percentile(pred['dcvi_score'], 33)
    p66 = np.percentile(pred['dcvi_score'], 66)
    pred['risk'] = pd.cut(pred['dcvi_score'], bins=[0, p33, p66, 100],
                          labels=['LOW','MEDIUM','HIGH'],
                          include_lowest=True).astype(str)

    d23  = model_df[model_df['year'] == 2023].copy()
    d23['district_name'] = d23['district'].map(dist_num_map)
    cats = d23[['district_name','crime_against_women','crime_against_children',
                'violent_crime','cyber_crime','other_crime']].set_index('district_name')
    pred = pred.merge(cats, left_on='district', right_index=True, how='left')

    hist = {}
    for dnum, dn in dist_num_map.items():
        rows = model_df[model_df['district']==dnum].sort_values('year')
        hist[dn] = rows[['year','total_crime']].set_index('year')['total_crime'].to_dict()

    return pred, hist, p33, p66


def load_metrics():
    default = {'r2': 0.776, 'mae': 205}
    if not os.path.exists('dcvi_metrics.json'):
        return default
    try:
        with open('dcvi_metrics.json', 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return {
            'r2': float(metrics.get('r2', default['r2'])),
            'mae': float(metrics.get('mae', default['mae']))
        }
    except Exception:
        return default

pred_df, hist_data, P33, P66 = load_data()
metrics = load_metrics()

COORDS = {
    "bagalkot":[16.18,75.69],"ballari":[15.15,76.93],
    "belagavi city":[15.85,74.50],"belagavi district":[15.70,74.90],
    "bengaluru city":[12.97,77.59],"bengaluru district":[13.10,77.45],
    "bidar":[17.91,77.52],"chamarajnagar":[11.92,76.94],
    "chikkaballapura":[13.44,77.73],"chikkamagaluru":[13.32,75.77],
    "chitradurga":[14.23,76.40],"dakshina kannada":[12.84,74.86],
    "davanagere":[14.46,75.92],"dharwad":[15.46,75.01],
    "gadag":[15.43,75.63],"hassan":[13.00,76.10],
    "haveri":[14.80,75.40],"hubballi dharwad city":[15.36,75.12],
    "k.g.f.":[12.97,78.27],"k.railways":[15.31,75.71],
    "kalaburgi":[17.33,76.83],"kalaburgi city":[17.40,76.90],
    "kodagu":[12.42,75.74],"kolar":[13.14,78.13],
    "koppal":[15.35,76.15],"mandya":[12.52,76.90],
    "mangaluru city":[12.91,74.85],"mysuru city":[12.30,76.65],
    "mysuru district":[12.20,76.40],"raichur":[16.20,77.35],
    "ramanagara":[12.72,77.28],"shimoga":[13.93,75.57],
    "tumakuru":[13.34,77.10],"udupi":[13.34,74.75],
    "uttara kannada":[14.79,74.13],"vijayanagara":[15.32,76.46],
    "vijayapura":[16.83,75.71],"yadgiri":[16.77,77.14],
}

RISK_CLR = {'HIGH':'#dc2626','MEDIUM':'#d97706','LOW':'#16a34a'}


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:38px'>🔴</div>
        <div style='font-size:26px;font-weight:700;margin-top:4px'>
            Karnataka Crime<br>Risk Dashboard
        </div>
    </div>
    <hr style='border-color:#45475a;margin:8px 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav", label_visibility="collapsed",
        options=["📍  District Prediction",
                 "🗺️  Bengaluru Hotspot Map",
                 "📊  All Districts Ranking"]
    )

    st.markdown("""
    <hr style='border-color:#45475a;margin:12px 0 8px'>
    <div style='font-size:15px;line-height:2.2'>
        <b>Model A</b> — DCVI Prediction<br>
        <b>Model B</b> — KDE Hotspot Map<br>
        <b>Data</b> — NCRB Karnataka 2020–23<br>
        <b>R2</b> = {r2:.3f} | <b>MAE</b> = {mae:.0f}<br>
        <b>Algorithm</b> — Random Forest
    </div>
    """.format(r2=metrics['r2'], mae=metrics['mae']), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 1 — DISTRICT PREDICTION
# ════════════════════════════════════════════════════════════
if page == "📍  District Prediction":

    st.markdown("## 📍 District Crime Risk Prediction")
    st.markdown("<div class='info-box'>Select a district and year to see DCVI score, "
                "risk level, predicted crime count, and category breakdown.</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([4, 1])
    with c1:
        district = st.selectbox("Select district",
                                sorted(pred_df['district'].tolist()),
                                format_func=lambda x: x.title())
    with c2:
        year = st.selectbox("Year", [2024, 2025])

    row         = pred_df[pred_df['district'] == district].iloc[0]
    risk        = str(row['risk'])
    pred_crimes = int(row['pred_crimes_2024'] if year == 2024 else row['pred_crimes_2025'])
    change_pct  = float(row['change_pct'])
    arrow       = "↑" if change_pct > 0 else "↓"

    # Metric row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("DCVI Score",          f"{row['dcvi_score']:.1f} / 100")
    m2.metric(f"Predicted {year}",    f"{pred_crimes:,}")
    m3.metric("Actual 2023",          f"{int(row['actual_2023']):,}")
    m4.metric("Change vs 2023",
              f"{arrow} {abs(change_pct):.1f}%",
              delta=f"{change_pct:+.1f}%",
              delta_color="inverse" if change_pct > 0 else "normal")
    with m5:
        st.markdown(
            f"<div style='margin-top:8px'>"
            f"<p style='font-size:355px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.05em;color:#6c757d;margin-bottom:16px'>RISK LEVEL</p>"
            f"<span class='badge badge-{risk}'>{risk}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='margin:18px 0 14px'>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<p class='sec-head'>Karnataka district map</p>",
                    unsafe_allow_html=True)

        m = folium.Map(location=[14.8, 76.5], zoom_start=6,
                       tiles="CartoDB positron")

        for _, r in pred_df.iterrows():
            d = str(r['district'])
            if d not in COORDS: continue
            lat, lon = COORDS[d]
            rc  = RISK_CLR.get(str(r['risk']), '#888')
            sel = (d == district)
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(9, r['dcvi_score'] / 7.5),
                color='#1e1e2e' if sel else rc,
                fill=True, fill_color=rc,
                fill_opacity=1.0 if sel else 0.72,
                weight=4 if sel else 1.5,
                tooltip=folium.Tooltip(
                    f"<b>{d.title()}</b><br>DCVI: {r['dcvi_score']:.1f}<br>Risk: {r['risk']}",
                    sticky=True)
            ).add_to(m)
            if sel:
                folium.Marker([lat+0.18, lon],
                    icon=folium.DivIcon(
                        html=f"<div style='font-size:25px;font-weight:700;"
                             f"color:#1e1e2e'>{d.title()}</div>",
                        icon_size=(160,20))
                ).add_to(m)

        st_folium(m, height=880, use_container_width=True, returned_objects=[])

    with right:
        st.markdown("<p class='sec-head'>Crime category breakdown — 2023 actual</p>",
                    unsafe_allow_html=True)

        cats   = ['Women','Children','Cyber']
        values = [int(row.get('crime_against_women',0)),
                  int(row.get('crime_against_children',0)),
                  int(row.get('cyber_crime',0)),]
        clrs   = ['#dc2626','#d97706','#7c3aed','#2563eb','#6b7280']

        fig_bar = go.Figure(go.Bar(
            x=cats, y=values, marker_color=clrs, marker_line_width=0,
            text=[f"{v:,}" for v in values],
            textposition='outside', textfont=dict(size=13, color='#374151')
        ))
        fig_bar.update_layout(
            margin=dict(t=20,b=10,l=10,r=10), height=400,
            showlegend=False, bargap=0.55,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='#f3f4f6',
                       tickfont=dict(size=13,color='#9ca3af')),
            xaxis=dict(showgrid=False, tickfont=dict(size=16,color='#374151'))
        )
        st.plotly_chart(fig_bar, use_container_width=True, key='cat_bar')

        st.markdown("<p class='sec-head'>Crime trend 2020–2025</p>",
                    unsafe_allow_html=True)

        hist     = hist_data.get(district, {})
        yrs_act  = [2020,2021,2022,2023]
        act_vals = [hist.get(y) for y in yrs_act]
        yrs_pred = [2023,2024,2025]
        prd_vals = [int(row['actual_2023']),
                    int(row['pred_crimes_2024']),
                    int(row['pred_crimes_2025'])]

        fig_ln = go.Figure()
        fig_ln.add_trace(go.Scatter(
            x=yrs_act, y=act_vals, mode='lines+markers', name='Actual',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=8, color='#2563eb', line=dict(color='white',width=2))
        ))
        fig_ln.add_trace(go.Scatter(
            x=yrs_pred, y=prd_vals, mode='lines+markers', name='Predicted',
            line=dict(color='#dc2626', width=3, dash='dash'),
            marker=dict(size=8, color='#dc2626', line=dict(color='white',width=2))
        ))
        fig_ln.add_vline(x=2023, line_dash='dot', line_color='#9ca3af',
                         annotation_text='Prediction starts',
                         annotation_font_size=20, annotation_font_color='#9ca3af',
                         annotation_position='top right')
        fig_ln.update_layout(
            margin=dict(t=10,b=10,l=10,r=10), height=400,
            showlegend=True,
            legend=dict(orientation='h', y=1.12, x=0,
                        font=dict(size=15), bgcolor='rgba(0,0,0,0)'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='#f3f4f6',
                       tickfont=dict(size=13,color='#9ca3af')),
            xaxis=dict(showgrid=False, dtick=1,
                       tickfont=dict(size=14,color='#374151'))
        )
        st.plotly_chart(fig_ln, use_container_width=True, key='trend_line')


# ════════════════════════════════════════════════════════════
#  PAGE 2 — BENGALURU HOTSPOT MAP
# ════════════════════════════════════════════════════════════
elif page == "🗺️  Bengaluru Hotspot Map":

    st.markdown("## 🗺️ Bengaluru Crime Hotspot Map")
    st.markdown(
        "<div class='info-box'>Model B — KDE (Kernel Density Estimation) on 6,303 real "
        "GPS crime incidents. Red = highest crime density. Use layer control (top-right) "
        "to toggle heatmap and individual crime markers.</div>",
        unsafe_allow_html=True
    )

    # Try all filenames the user might have created
    possible = [
        "bengaluru_hotspot_all.html",
        "bengaluru_hotspot_murder.html",
        "karnataka_crime_2024_fixed.html",
        "karnataka_crime_2024.html",
        "final_crime_heatmap.html",
        "bengaluru_hotspot_all_crimes.html",
    ]

    html_file = next((f for f in possible if os.path.exists(f)), None)

    if html_file:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.success(f"✅ Loaded: **{html_file}**")
        st.components.v1.html(html_content, height=650, scrolling=False)
    else:
        st.warning("⚠️ Bengaluru hotspot HTML not found. Run `python kar_map_fixed.py` "
                   "then refresh.")
        st.markdown("**The app looks for any of these files:**")
        for f in possible:
            st.code(f)
        # Placeholder map
        ph = folium.Map(location=[12.9716,77.5946], zoom_start=7,
                        tiles='CartoDB dark_matter')
        folium.Marker([13.0716,77.1946],
                      tooltip="Run kar_map_fixed.py to load real hotspot data"
                      ).add_to(ph)
        st_folium(ph, height=800, use_container_width=True, returned_objects=[])


# ════════════════════════════════════════════════════════════
#  PAGE 3 — ALL DISTRICTS RANKING
# ════════════════════════════════════════════════════════════
elif page == "📊  All Districts Ranking":

    st.markdown("## 📊 All 38 Karnataka Districts — DCVI Rankings")
    st.markdown(
        "<div class='info-box'>DCVI score (0–100) combines crime volume (35%), "
        "women+children ratio (35%), trend (20%), and crime variety (10%). "
        "Higher score = higher vulnerability.</div>",
        unsafe_allow_html=True
    )

    year_sel = st.radio("Prediction year", [2024, 2025], horizontal=True)
    col_pred = 'pred_crimes_2024' if year_sel == 2024 else 'pred_crimes_2025'

    # Create display dataframe with formatted values
    display_df = pred_df[['district','dcvi_score','risk',
                           col_pred,'actual_2023','change_pct']].copy()
    display_df.columns = ['District','DCVI Score','Risk',
                           f'Predicted {year_sel}','Actual 2023','Change %']
    display_df['District'] = display_df['District'].str.title()
    display_df = display_df.sort_values('DCVI Score', ascending=False).reset_index(drop=True)
    display_df.insert(0, 'Rank', range(1, len(display_df)+1))

    # Create formatted display version for table
    display_df_formatted = display_df.copy()
    display_df_formatted['DCVI Score'] = display_df_formatted['DCVI Score'].round(1).astype(str)
    display_df_formatted[f'Predicted {year_sel}'] = display_df_formatted[f'Predicted {year_sel}'].apply(lambda x: f"{int(x):,}")
    display_df_formatted['Actual 2023'] = display_df_formatted['Actual 2023'].apply(lambda x: f"{int(x):,}")
    display_df_formatted['Change %'] = display_df_formatted['Change %'].apply(lambda x: f"{x:+.1f}%")

    display_style = display_df_formatted.style.hide(axis='index').set_table_styles([
        {'selector': 'th', 'props': [('font-size', '32px'), ('padding', '16px 18px'), ('text-align','left')]},
        {'selector': 'td', 'props': [('font-size', '32px'), ('padding', '16px 18px'), ('text-align','left')]},
        {'selector': 'thead th', 'props': [('font-size', '34px'), ('font-weight', '700')]},
        {'selector': 'tbody tr', 'props': [('height', '56px')]}
    ])
    st.dataframe(display_style, use_container_width=True, height=560)

    st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("<p class='sec-head'>All districts ranked by DCVI score</p>",
                unsafe_allow_html=True)

    # Keep numeric values for charting
    chart_df = display_df.sort_values('DCVI Score', ascending=True)
    fig_all  = px.bar(
        chart_df, x='DCVI Score', y='District',
        color='Risk',
        color_discrete_map={'HIGH':'#dc2626','MEDIUM':'#d97706','LOW':'#16a34a'},
        orientation='h', height=700, text='DCVI Score'
    )
    fig_all.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside',
        textfont=dict(size=15, color='#374151'),
        marker_line_width=0
    )
    fig_all.update_layout(
        margin=dict(t=20,b=20,l=10,r=80),
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0,
                    font=dict(size=18), title=None),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#f3f4f6',
                   tickfont=dict(size=16,color='#9ca3af')),
        yaxis=dict(showgrid=False, tickfont=dict(size=13,color='#374151'))
    )
    fig_all.add_vline(x=P33, line_dash='dash', line_color='#16a34a', line_width=1.7,
                      annotation_text=f"LOW/MEDIUM ({P33:.0f})",
                      annotation_font_size=10, annotation_font_color='#16a34a',
                      annotation_position='top right')
    fig_all.add_vline(x=P66, line_dash='dash', line_color='#dc2626', line_width=1.7,
                      annotation_text=f"MEDIUM/HIGH ({P66:.0f})",
                      annotation_font_size=10, annotation_font_color='#dc2626',
                      annotation_position='top right')
    st.plotly_chart(fig_all, use_container_width=True, key='all_bar')

    st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("<p class='sec-head'>Summary</p>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("High risk districts",   str((pred_df['risk']=='HIGH').sum()))
    s2.metric("Medium risk districts", str((pred_df['risk']=='MEDIUM').sum()))
    s3.metric("Low risk districts",    str((pred_df['risk']=='LOW').sum()))
    s4.metric("Avg DCVI score",        f"{pred_df['dcvi_score'].mean():.1f}")