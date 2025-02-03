import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from io import BytesIO

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Gadamer007/Retirement_location/main/Ret_data.xlsx"  # GitHub raw URL
    
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
continent_mapping = { ... }  # Same mapping as before

data['Continent'] = data['Country'].map(continent_mapping)

# Categorize each variable into percentiles
def categorize_percentiles(df, variables):
    for var in variables:
        if var in df.columns:
            if var == "Pollution":
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='min', ascending=False, na_option='bottom'),
                    5, labels=[5, 4, 3, 2, 1]
                )
            else:
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='first', ascending=True, na_option='bottom'),
                    5, labels=[5, 4, 3, 2, 1]
                )
    return df

data = categorize_percentiles(data, list(column_mapping.values()))

# Title
st.title("Best Countries for Early Retirement: Where to Retire Abroad?")

# Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
selected_vars = []
sliders = {}
variables = list(column_mapping.values())

for label in variables:
    if st.sidebar.checkbox(label, value=True):
        sliders[label] = st.sidebar.slider(f"{label} (1=Best, 5=Worst)", 1, 5, 5, format=None)
        selected_vars.append(label)

if selected_vars:
    df_selected = data[['Country', 'Col_2025', 'Continent'] + selected_vars + [f"{var}_Category" for var in selected_vars]]
    df_selected['Available Count'] = df_selected[selected_vars].notna().sum(axis=1)
    df_selected = df_selected.dropna(subset=selected_vars, how='all')
    
    for var in selected_vars:
        max_category = sliders[var]  
        df_selected = df_selected[df_selected[f"{var}_Category"].astype(int) <= max_category]
    
    df_selected['Retirement Suitability'] = df_selected[selected_vars].mean(axis=1, skipna=True)
    
    # Preserve continent selection
    if 'continent_selection' not in st.session_state:
        st.session_state['continent_selection'] = df_selected['Continent'].unique().tolist()
    
    df_selected = df_selected[df_selected['Continent'].isin(st.session_state['continent_selection'])]
    
    # Scatter Plot
    fig_scatter = px.scatter(
        df_selected, 
        x="Retirement Suitability", 
        y="Col_2025", 
        text="Country", 
        color=df_selected['Continent'],
        title="Retirement Suitability vs Cost of Living", 
        labels={
            "Col_2025": "Cost of Living (0 - 100)", 
            "Retirement Suitability": "Retirement Suitability (0 - 100)"
        },
        template="plotly_dark", 
        category_orders={"Continent": ["America", "Europe", "Asia", "Africa", "Oceania"]},
        hover_data={var: ':.2f' for var in selected_vars}
    )
    
    # Add red circles around incomplete data
    incomplete_data = df_selected[df_selected['Available Count'] < len(selected_vars)]
    fig_scatter.add_trace(
        px.scatter(
            incomplete_data,
            x="Retirement Suitability", 
            y="Col_2025", 
            text="Country", 
            marker=dict(symbol='circle-open', color='red', size=12),
            hovertext="Incomplete Data"
        ).data[0]
    )
    
    fig_scatter.update_layout(
        legend_title_text="Continent",
        legend_traceorder="grouped"
    )
    
    # Allow user to filter continents without resetting
    selected_continents = st.multiselect("Select Continents", options=df_selected['Continent'].unique(), default=st.session_state['continent_selection'])
    st.session_state['continent_selection'] = selected_continents
    df_selected = df_selected[df_selected['Continent'].isin(selected_continents)]
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Map Visualization
    st.write("### Understand the spatial distribution of the variables that make up the Retirement Suitability")
    selected_map_var = st.selectbox("", selected_vars)
    
    fig_map = px.choropleth(df_selected, locations="Country", locationmode="country names", color=selected_map_var, color_continuous_scale="RdYlGn")
    
    st.plotly_chart(fig_map, use_container_width=True)

