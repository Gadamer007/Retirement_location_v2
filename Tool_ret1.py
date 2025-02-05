import streamlit as st
import pandas as pd
import plotly.express as px
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

# Define the retirement suitability variables
selected_vars = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate",
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]

# Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
user_selected_vars = []
sliders = {}

for label in selected_vars:
    if st.sidebar.checkbox(label, value=True):
        sliders[label] = st.sidebar.slider(f"{label}", 1, 5, 5)
        user_selected_vars.append(label)

# Ensure valid selections
if user_selected_vars:
    df_filtered = data.copy()

    # Apply slider filters
    for var in user_selected_vars:
        if var in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[var].astype(float) <= sliders[var]]

    # Ensure at least one valid variable exists
    available_vars = [var for var in user_selected_vars if var in df_filtered.columns]
    if not available_vars:
        st.error("No valid variables selected. Please check the dataset or adjust your selections.")
        st.stop()

    # Create df_selected with available variables
    df_selected = df_filtered[['Country', 'Cost of Living', 'Continent'] + available_vars].copy()

    # Compute Retirement Suitability Score
    df_selected['Retirement Suitability'] = df_selected[available_vars].astype(float).mean(axis=1)

    # Plot Scatter Chart
    fig_scatter = px.scatter(
        df_selected,
        x="Retirement Suitability",
        y="Cost of Living",
        text="Country",
        color="Continent",
        title="Retirement Suitability vs Cost of Living",
        labels={"Cost of Living": "Cost of Living (0 - 100)", "Retirement Suitability": "Retirement Suitability (0 - 100)"},
        template="plotly_dark",
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Plot Map
    selected_map_var = st.selectbox("Select a Variable to Map", available_vars)
    fig_map = px.choropleth(df_selected, locations="Country", locationmode="country names", color=selected_map_var, title=f"{selected_map_var} by Country")
    st.plotly_chart(fig_map, use_container_width=True)








