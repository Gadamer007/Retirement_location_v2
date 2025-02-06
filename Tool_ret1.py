import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from io import BytesIO

# 🎨 Sidebar Styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] { min-width: 360px !important; max-width: 360px !important; }
    [data-testid="stSidebar"] label { font-size: 11px !important; font-weight: 500 !important; margin-bottom: -10px !important; white-space: nowrap !important; }
    [data-testid="stSidebar"] .st-bb { margin-bottom: -15px !important; }
    [data-testid="stSidebar"] .st-br { margin-bottom: 10px !important; }
    [data-testid="stSidebarContent"] { padding-right: 15px !important; }
    </style>
    """, unsafe_allow_html=True)

# 🎯 Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gadamer007/Retirement_location_v2/main/Ret_data.xlsx"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Country")
    return df

# 📌 Load and Preprocess Data
data = load_data()
data["Country"] = data["Country"].str.strip().str.title()

# 🌎 Define Country-to-Continent Mapping
continent_mapping = {
    'United States': 'America', 'Canada': 'America', 'Mexico': 'America', 'Brazil': 'America',
    'Argentina': 'America', 'Chile': 'America', 'Colombia': 'America', 'Peru': 'America', 'Uruguay': 'America',
    'Costa Rica': 'America', 'Panama': 'America', 'Trinidad And Tobago': 'America', 'Puerto Rico': 'America',
    'Dominican Republic': 'America', 'Paraguay': 'America', 'Ecuador': 'America', 'Venezuela': 'America',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa', 'Morocco': 'Africa',
    'Mauritius': 'Africa', 'Tunisia': 'Africa', 'Ghana': 'Africa', 'Uganda': 'Africa', 'Algeria': 'Africa',
    'Libya': 'Africa', 'Zimbabwe': 'Africa', 'Iceland': 'Europe', 'Hong Kong (China)': 'Asia',
    'China': 'Asia', 'Japan': 'Asia', 'India': 'Asia', 'South Korea': 'Asia', 'Thailand': 'Asia',
    'Singapore': 'Asia', 'Malaysia': 'Asia', 'Israel': 'Asia', 'Taiwan': 'Asia', 'Jordan': 'Asia',
    'Kazakhstan': 'Asia', 'Lebanon': 'Asia', 'Armenia': 'Asia', 'Iraq': 'Asia', 'Uzbekistan': 'Asia',
    'Vietnam': 'Asia', 'Philippines': 'Asia', 'Kyrgyzstan': 'Asia', 'Bangladesh': 'Asia', 'Iran': 'Asia',
    'Nepal': 'Asia', 'Sri Lanka': 'Asia', 'Pakistan': 'Asia', 'Kuwait': 'Asia', 'Turkey': 'Asia',
    'Indonesia': 'Asia', 'United Arab Emirates': 'Asia', 'Saudi Arabia': 'Asia', 'Bahrain': 'Asia',
    'Qatar': 'Asia', 'Oman': 'Asia', 'Azerbaijan': 'Asia',
    'Germany': 'Europe', 'France': 'Europe', 'United Kingdom': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
    'Netherlands': 'Europe', 'Sweden': 'Europe', 'Denmark': 'Europe', 'Norway': 'Europe', 'Ireland': 'Europe',
    'Finland': 'Europe', 'Belgium': 'Europe', 'Austria': 'Europe', 'Switzerland': 'Europe', 'Luxembourg': 'Europe',
    'Czech Republic': 'Europe', 'Slovenia': 'Europe', 'Estonia': 'Europe', 'Poland': 'Europe', 'Malta': 'Europe',
    'Croatia': 'Europe', 'Lithuania': 'Europe', 'Slovakia': 'Europe', 'Latvia': 'Europe', 'Portugal': 'Europe',
    'Bulgaria': 'Europe', 'Hungary': 'Europe', 'Romania': 'Europe', 'Greece': 'Europe', 'Montenegro': 'Europe',
    'Serbia': 'Europe', 'Bosnia And Herzegovina': 'Europe', 'North Macedonia': 'Europe', 'Albania': 'Europe',
    'Moldova': 'Europe', 'Belarus': 'Europe', 'Georgia': 'Europe', 'Ukraine': 'Europe', 'Russia': 'Europe',
    'Cyprus': 'Europe', 'Kosovo (Disputed Territory)': 'Europe', 'Australia': 'Oceania', 'New Zealand': 'Oceania'
}

# 🔥 Assign Continent
data["Continent"] = data["Country"].map(continent_mapping)

# 📊 Variables
variables = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate", 
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]

# 🚀 Normalize & Rank Data
def normalize_and_rank(df):
    for var in variables:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors="coerce")
            
            # Normalize 0-100
            df[var] = df[var].rank(pct=True) * 100  
            
            # Reverse scale for "lower is better" variables
            if var in ["Pollution", "Openness", "Natural Scenery", "Natural Disaster"]:
                df[var] = 100 - df[var]

            # Rank Categories (5 = Worst, 1 = Best)
            df[f"{var}_Category"] = pd.qcut(df[var], q=5, labels=[5, 4, 3, 2, 1])
    
    return df

data = normalize_and_rank(data)

# 🛠️ Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
sliders = {var: st.sidebar.slider(var, 1, 5, 5) for var in variables}

# 🎯 Apply Filters
df_filtered = data.copy()
for var in variables:
    df_filtered = df_filtered[df_filtered[f"{var}_Category"].fillna(5).astype(int) <= sliders[var]]

# 📌 Multi-Select Continent Filter
selected_continents = st.sidebar.multiselect("Select Continents", df_filtered["Continent"].unique(), df_filtered["Continent"].unique())
df_filtered = df_filtered[df_filtered["Continent"].isin(selected_continents)]

# 🏆 Compute Suitability Score
df_filtered["Retirement Suitability"] = df_filtered[variables].mean(axis=1, skipna=True)

# 📈 Plot
if df_filtered.empty:
    st.error("No data available to plot. Adjust filter settings.")
else:
    fig = px.scatter(
        df_filtered, x="Retirement Suitability", y="Cost of Living", text="Country",
        color="Continent", title="Retirement Suitability vs Cost of Living",
        labels={"Cost of Living": "Cost of Living (0 - 100)", "Retirement Suitability": "Retirement Suitability (0 - 100)"},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)








