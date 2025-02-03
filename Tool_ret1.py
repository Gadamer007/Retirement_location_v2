import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from io import BytesIO

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gadamer007/Retirement_location/main/Ret_data.xlsx"  
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Country")
    return df

# Load the data
data = load_data()
data['Country'] = data['Country'].str.strip().str.title()

# Standardize column names
column_mapping = {
    "Safety index_2025": "Safety",
    "Healthcare_2025": "Healthcare",
    "Political stability_2023": "Political Stability",
    "Pollution_2025": "Pollution",
    "Climate_2025": "Climate"
}
data = data.rename(columns=column_mapping)

# Define country-to-continent mapping
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

# Assign continent
data['Continent'] = data['Country'].map(continent_mapping)

# Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
selected_vars = []
variables = list(column_mapping.values())

for label in variables:
    if st.sidebar.checkbox(label, value=True):
        selected_vars.append(label)

if selected_vars:
    df_filtered = data.copy()
    df_selected = df_filtered[['Country', 'Col_2025', 'Continent'] + selected_vars].copy()
    df_selected['Valid_Var_Count'] = df_selected[selected_vars].count(axis=1)
    df_selected['Retirement Suitability'] = df_selected[selected_vars].mean(axis=1)

    incomplete_data = df_selected[df_selected['Valid_Var_Count'] < len(selected_vars)]
    
    df_selected["Data_Completion"] = df_selected["Country"].apply(
        lambda x: "Incomplete Data" if x in incomplete_data["Country"].values else "Complete Data"
    )

    # Create figure object
    fig = go.Figure()

    # 🌍 First trace (Color Legend - Continents)
    for continent in df_selected["Continent"].unique():
        subset = df_selected[df_selected["Continent"] == continent]
        fig.add_trace(go.Scatter(
            x=subset["Retirement Suitability"],
            y=subset["Col_2025"],
            mode="markers",
            marker=dict(size=12, color=subset["Continent"].map({
                "Europe": "blue", "America": "red", "Asia": "green", "Oceania": "purple", "Africa": "orange"
            })),
            name=continent,
            legendgroup="Continent",  # Group all continents together
        ))

    # ✅ Second trace (Shape Legend - Data Completeness)
    for completeness in ["Complete Data", "Incomplete Data"]:
        subset = df_selected[df_selected["Data_Completion"] == completeness]
        fig.add_trace(go.Scatter(
            x=subset["Retirement Suitability"],
            y=subset["Col_2025"],
            mode="markers",
            marker=dict(size=12, symbol="circle" if completeness == "Complete Data" else "x", 
                        color="black"),  # Keep color from continents above
            name=completeness,
            legendgroup="Data Availability",  # Separate legend
        ))

    # Update layout for **two separate legends**
    fig.update_layout(
        title="Retirement Suitability vs Cost of Living",
        xaxis_title="Retirement Suitability (0 - 100)",
        yaxis_title="Cost of Living (0 - 100)",
        legend=dict(traceorder="grouped"),
        paper_bgcolor='black', 
        plot_bgcolor='black'
    )

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)






