import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from io import BytesIO

# 🔹 Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gadamer007/Retirement_location_v2/main/Ret_data.xlsx"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Country")
    return df

# Load the data
data = load_data()
data['Country'] = data['Country'].str.strip().str.title()

# 🔹 Define Variables
variables = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate",
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]

high_is_better = ["Safety", "Healthcare", "Political Stability", "Climate", "English Proficiency"]
low_is_better = ["Pollution", "Openness", "Natural Scenery"]

# 🔹 Normalize & Categorize Data
def normalize_and_rank(df):
    for var in variables:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df[var] = df[var].replace([np.inf, -np.inf], np.nan)

            # Normalize all variables (0-100)
            df[var] = (df[var] - df[var].min()) / (df[var].max() - df[var].min()) * 100

            # Reverse scoring for low-is-better variables
            if var in low_is_better:
                df[var] = 100 - df[var]

            # Rank-normalize into 5 bins (20% each)
            try:
                df[f"{var}_Category"] = pd.qcut(df[var], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop")
            except ValueError:
                df[f"{var}_Category"] = pd.cut(df[var], bins=5, labels=[5, 4, 3, 2, 1], include_lowest=True)

            # 🔥 Fix NaN issue before converting to int
            df[f"{var}_Category"] = df[f"{var}_Category"].astype("category").cat.add_categories(5).fillna(5).astype(int)

    return df

data = normalize_and_rank(data)

# 🔹 Sidebar Controls
st.sidebar.subheader("Select Variables for Retirement Suitability")
selected_vars = [var for var in variables if st.sidebar.checkbox(var, value=True)]
sliders = {var: st.sidebar.slider(var, 1, 5, 5) for var in selected_vars}

# 🔹 Filter Data by Slider Selections
df_filtered = data.copy()
for var in selected_vars:
    df_filtered = df_filtered[df_filtered[f"{var}_Category"] <= sliders[var]]

# 🔹 Compute Suitability Score
df_filtered["Retirement Suitability"] = df_filtered[selected_vars].mean(axis=1)

# 🔹 Continent Filter
selected_continents = st.multiselect("Select Continents", df_filtered["Continent"].unique(), df_filtered["Continent"].unique())
df_filtered = df_filtered[df_filtered["Continent"].isin(selected_continents)]

# 🔹 Scatter Plot
fig = px.scatter(
    df_filtered, x="Retirement Suitability", y="Cost of Living", text="Country", color="Continent",
    title="Retirement Suitability vs Cost of Living",
    labels={"Cost of Living": "Cost of Living (0-100)", "Retirement Suitability": "Retirement Suitability (0-100)"},
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)









