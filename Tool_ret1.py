import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from io import BytesIO

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gadamer007/Retirement_location_v2/main/Ret_data.xlsx"  # Corrected GitHub repo
    
    # Download the file content from GitHub
    response = requests.get(url)
    response.raise_for_status()  # Stop if the request fails

    # Load the Excel file into Pandas from the downloaded content
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

# Title for the Tool
st.title("Best Countries for Early Retirement: Where to Retire Abroad?")

# Sidebar Filters
# Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
selected_vars = []
sliders = {}

# Define the variables available for selection
variables = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate", 
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]

# Function to categorize each variable into 5 percentiles
def categorize_percentiles(df, variables):
    for var in variables:
        if var in df.columns:
            if var == "Pollution":
                # Invert Pollution so lower values are better
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='min', ascending=False, na_option='bottom'),
                    5, labels=[5, 4, 3, 2, 1]  # 1 = Cleanest, 5 = Most Polluted
                )
            else:
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='min', ascending=True, na_option='bottom'),
                    5, labels=[1, 2, 3, 4, 5]  # 1 = Best, 5 = Worst
                )
    return df

# Apply rank normalization to the dataset
data = categorize_percentiles(data, variables)

# Create sidebar checkboxes and sliders using rank categories
for label in variables:
    category_col = f"{label}_Category"
    if category_col in data.columns:
        if st.sidebar.checkbox(label, value=True):
            sliders[label] = st.sidebar.slider(
                f"{label}", 
                1, 5, 5  # Always use 1-5 scale
            )
            selected_vars.append(label)





if not selected_vars:
    st.error("⚠️ No valid variables selected. Please check the dataset or adjust your selections.")
    st.stop()

# Start with full dataset
# Start with full dataset
df_filtered = data.copy()

# Apply slider filters based on rank categories
for var in selected_vars:
    max_category = sliders[var]
    category_col = f"{var}_Category"
    df_filtered = df_filtered[df_filtered[category_col].astype(int) <= max_category]

# Ensure selected variables exist
available_vars = [var for var in selected_vars if f"{var}_Category" in df_filtered.columns]

if not available_vars:
    st.error("⚠️ No valid variables after filtering. Adjust your selections.")
    st.stop()

df_selected = df_filtered[['Country', 'Cost of Living', 'Continent'] + [f"{var}_Category" for var in available_vars]].copy()

# Compute Retirement Suitability Score
df_selected['Retirement Suitability'] = df_selected[selected_vars].mean(axis=1)

# Multi-select continent filter
selected_continents = st.multiselect(
    "Select Continents to Display", 
    options=df_selected["Continent"].unique().tolist(), 
    default=df_selected["Continent"].unique().tolist()
)

df_selected = df_selected[df_selected["Continent"].isin(selected_continents)]

# Scatter Plot
fig_scatter = px.scatter(
    df_selected, 
    x="Retirement Suitability", 
    y="Cost of Living", 
    text="Country", 
    color=df_selected['Continent'],
    title="Retirement Suitability vs Cost of Living", 
    labels={
        "Cost of Living": "Cost of Living (0 - 100)", 
        "Retirement Suitability": "Retirement Suitability (0 - 100)"
    },
    template="plotly_dark", 
    category_orders={"Continent": ["America", "Europe", "Asia", "Africa", "Oceania"]},
    hover_data={var: ':.2f' for var in selected_vars}
)

fig_scatter.update_traces(marker=dict(size=10), textposition='top center')
fig_scatter.update_layout(
    title=dict(text="Retirement Suitability vs Cost of Living", font=dict(color='white', size=24), x=0.5, xanchor="center"),
    xaxis=dict(linecolor='white', tickfont=dict(color='white'), showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1),
    yaxis=dict(linecolor='white', tickfont=dict(color='white'), showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1),
    legend=dict(font=dict(color="white")),
    paper_bgcolor='black', plot_bgcolor='black'
)

st.plotly_chart(fig_scatter, use_container_width=True)






