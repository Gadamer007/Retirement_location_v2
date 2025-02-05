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
# Function to normalize and categorize variables into 5 rank groups
# Function to normalize and categorize variables into 5 rank groups
def categorize_percentiles(df, variables):
    for var in variables:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')  # Ensure numeric
            df[var] = df[var].fillna(df[var].median())  # Fill NaN with median

            # Normalize all variables to a 0-100 scale
            min_val = df[var].min()
            max_val = df[var].max()
            if min_val != max_val:  # Avoid division by zero
                df[var] = (df[var] - min_val) / (max_val - min_val) * 100

            # Special handling for English Proficiency (convert categorical 1-5 to meaningful values)
            if var == "English Proficiency":
                proficiency_mapping = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
                df[var] = df[var].map(proficiency_mapping)

            # Try using qcut first, if it fails use cut
            try:
                if var == "Pollution":  # Pollution is inverse (higher is worse)
                    df[f"{var}_Category"] = pd.qcut(
                        df[var].rank(method='min', ascending=False, na_option='bottom'),
                        q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
                    )
                else:
                    df[f"{var}_Category"] = pd.qcut(
                        df[var].rank(method='min', ascending=True, na_option='bottom'),
                        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
                    )
            except ValueError:
                # Fallback to pd.cut if qcut fails due to duplicate edges
                df[f"{var}_Category"] = pd.cut(
                    df[var].rank(method='min', ascending=True, na_option='bottom'),
                    bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True
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

# Ensure selected variables exist and retrieve their actual values (0-100)
real_value_vars = [var for var in selected_vars if var in data.columns]

if not real_value_vars:
    st.error("⚠️ No valid variables after filtering. Adjust your selections.")
    st.stop()

# Create df_selected using real values, not just categories
df_selected = df_filtered[['Country', 'Cost of Living', 'Continent']].copy()

# Add selected variables' real values (0-100)
for var in real_value_vars:
    if var in data.columns:
        df_selected[var] = data[var]  # Store real values instead of categories

# Compute Retirement Suitability Score using actual values
df_selected['Retirement Suitability'] = df_selected[real_value_vars].mean(axis=1)


# Add the selected variables' real values (0-100) to the dataframe
for var in real_value_vars:
    df_selected[var] = data[var]  # Add actual values, not categories

# Compute Retirement Suitability Score using actual values
if real_value_vars:
    df_selected['Retirement Suitability'] = df_selected[real_value_vars].mean(axis=1)
else:
    df_selected['Retirement Suitability'] = np.nan  # Avoids errors if no variables are selected

# Compute Retirement Suitability Score using actual values (0-100), not rank categories
real_value_vars = [var for var in selected_vars if var in df_selected.columns]

if not real_value_vars:
    st.error("⚠️ No valid variables found for Retirement Suitability Score calculation.")
    st.stop()

# Compute Retirement Suitability Score using actual values
df_selected['Retirement Suitability'] = df_selected[real_value_vars].astype(float).mean(axis=1)



# Multi-select continent filter
selected_continents = st.multiselect(
    "Select Continents to Display", 
    options=df_selected["Continent"].unique().tolist(), 
    default=df_selected["Continent"].unique().tolist()
)

df_selected = df_selected[df_selected["Continent"].isin(selected_continents)]

# Scatter Plot
# Ensure correct hover data references to _Category columns
hover_data_adjusted = {f"{var}_Category": ':.2f' for var in selected_vars if f"{var}_Category" in df_selected.columns}

# Update hover data to show actual values instead of 1-5 categories
hover_data_adjusted = {var: ':.2f' for var in real_value_vars}

# Fix hover data: Use real values (0-100), not categories
hover_data_adjusted = {var: ':.2f' for var in real_value_vars}

fig_scatter = px.scatter(
    df_selected, 
    x="Retirement Suitability",  # ✅ Uses real values (0-100), not rank categories
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
    hover_data=hover_data_adjusted  # ✅ Shows actual values in hover
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





