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
# 🚀 Temporarily disable caching to force a fresh dataset reload
def load_data():
    url = "https://raw.githubusercontent.com/Gadamer007/Retirement_location_v2/main/Ret_data.xlsx"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="Country")
    return df

# 📌 Load and Preprocess Data
data = load_data()
# st.write(data[data["Country"] == "Hong Kong (China)"])
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
    # ✅ NEW: Added Cost of Living as an available variable
]

map_variables = variables + ["Cost of Living"]  # ✅ NEW: Separate list for the map



# 🚀 Normalize & Rank Data
def normalize_and_rank(df):
    for var in variables:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors="coerce")
            df[var] = df[var].rank(pct=True) * 100  

            if var in ["Pollution", "Openness", "Natural Scenery", "Natural Disaster"]:
                df[var] = 100 - df[var]  

            df[f"{var}_Category"] = pd.qcut(df[var], q=5, labels=[5, 4, 3, 2, 1])  

    return df

data = normalize_and_rank(data)

# 🏆 Title at the Very Top
st.title("Best Countries for Early Retirement: Where to Retire Abroad?")

# Instructions Section
with st.expander("📖 Instructions (Click to Expand/Collapse)"):
    st.write("""
    - This tool helps users identify the **best countries for retirement** abroad.  
    - Use the **left panel** to **select variables** that matter to you (e.g., uncheck Pollution if it’s not a factor).  
    - **Adjust sliders** to filter out low-performing countries for a given variable.  
    - The tool calculates a **Retirement Suitability Score** as the average of the selected factors.  
    - The **figure plots this score** against **cost of living (COL)** to highlight potential destinations.  
    - Use the **zoom tool** (top-right of the figure) to explore specific countries.  
    - Select/unselect **Continents** of interest.  
    - If desired, remove countries with **missing data** directly above the plot.
    - The **map below** shows how the Retirement Suitability factors are distributed geographically.   
    - The tool does not account for **Capital Gains Tax (CGT)**, but users should consider it **[see here](https://taxsummaries.pwc.com/quick-charts/capital-gains-tax-cgt-rates)**.  
    """)
st.write(instructions)

# 🌎 Continent Selection Directly Below Title
st.subheader("🌍 Select Continents to Display")
selected_continents = st.multiselect(
    "Select Continents",
    options=data["Continent"].unique().tolist(),
    default=data["Continent"].unique().tolist()
)
df_filtered = data[data["Continent"].isin(selected_continents)]

# 🚫 Checkbox to Exclude Incomplete Data
exclude_incomplete = st.checkbox("Exclude countries with incomplete data (more than 2 N/A)")
if exclude_incomplete:
    df_filtered = df_filtered[df_filtered.isna().sum(axis=1) <= 2]

# 🛠️ Sidebar Filters (Restored Sliders)
st.sidebar.subheader("Select Variables for Retirement Suitability")
st.sidebar.markdown("<small>Move slider to the left to drop worst performing countries. For example, moving slider from 5 to 4 drops the bottom 20% performing countries.</small>", unsafe_allow_html=True)
sliders = {}
selected_vars = []

cols = st.sidebar.columns(2)  
for i, var in enumerate(variables):
    with cols[i % 2]:  
        checked = st.checkbox(f"**{var}**", value=True, key=f"check_{var}")
        if checked:
            sliders[var] = st.slider("", 1, 5, 5, key=f"slider_{var}")
            selected_vars.append(var)

# 🎯 Apply Filters
for var in selected_vars:
    df_filtered = df_filtered[
        (df_filtered[f"{var}_Category"].fillna(1).astype(int) <= sliders[var])  # ✅ Keep NAs
        | (df_filtered[f"{var}_Category"].isna())  # ✅ Allow missing values to pass through
    ]


# 🏆 Compute Suitability Score
df_filtered["Retirement Suitability"] = df_filtered[selected_vars].drop(columns=["Cost of Living"], errors="ignore").mean(axis=1, skipna=True)

# 📈 Plot
if df_filtered.empty:
    st.error("No data available to plot. Adjust filter settings.")
else:
    # 📌 Define hover information (Ensures all 9 variables appear with NA for missing)
    # 🔧 Fix missing values (Convert NaN to "NA" explicitly before passing to hover data)
    # 🔧 Convert numeric columns to float (ensures fillna("NA") doesn't cause TypeError)
    for col in df_filtered.columns:
        if df_filtered[col].dtype == "object":  # Apply only to non-numeric columns
            df_filtered[col] = df_filtered[col].fillna("NA")
        else:
            df_filtered[col] = df_filtered[col].astype(float).round(2)  # ✅ Round decimals to 2

    
   
    # 📌 Define hover_data explicitly to avoid conflicts with data_frame
    # 📌 Ensure hover data explicitly avoids duplicate column conflicts
    hover_data = {
        "Continent": True,
        "Country": True,
        "Retirement Suitability": ":.2f",
        "Cost of Living": ":.2f",
    }
    
    # ➕ Manually add selected variables without conflicting with the DataFrame
    # Fix: Ensure all variables exist in df_filtered before adding to hover_data
    for var in variables:
        if var in df_filtered.columns:
            df_filtered[var] = pd.to_numeric(df_filtered[var], errors="coerce").round(2)  # ✅ Ensure numeric type
            hover_data[f"{var} (0-100)"] = df_filtered[var].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")  # ✅ Rename variable in hover_data to prevent conflicts


    
    # 📈 Updated Scatter Plot
    # 🌍 Define Fixed Colors for Continents
    continent_colors = {
        "America": "#ff7f0e",   # Blue
        "Europe": "#1f77b4",    # Orange 
        "Asia": "#2ca02c",      # Green
        "Africa": "#d62728",    # Red
        "Oceania": "#9467bd"    # Purple
    }
    
    # 📌 Updated Scatter Plot with Fixed Colors
    fig = px.scatter(
        df_filtered, 
        x="Retirement Suitability", 
        y="Cost of Living", 
        text="Country",
        color="Continent", 
        title="Retirement Suitability vs Cost of Living",
        labels={
            "Cost of Living": "Cost of Living (0 - 100)", 
            "Retirement Suitability": "Retirement Suitability (0 - 100)"
        },
        template="plotly_dark",
        hover_data=hover_data,  
        category_orders={"Continent": list(continent_colors.keys())},  # ✅ Keeps order consistent
        color_discrete_map=continent_colors  # ✅ Fixes colors
    )




    
    # ✅ Bubble size reduction and text over bubbles
    fig.update_traces(marker=dict(size=8), textposition="top center")
    
    # ✅ Dark background and gridline styling
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        title=dict(
            text="Retirement Suitability vs Cost of Living",  # ✅ Explicitly set title
            font=dict(color="white", size=26),  # ✅ Increase font size (default ~24, so +2 units)
            x=0.5,  # ✅ Center title horizontally
            xanchor="center"  # ✅ Ensure title stays centered
        ),

        xaxis=dict(
            showgrid=True,  # ✅ Enables vertical grid lines
            gridcolor="rgba(255, 255, 255, 0.3)",  # ✅ Matches horizontal grid color
            linecolor="white",
            tickfont=dict(color="white")
        ),
        yaxis=dict(
            showgrid=True,  # ✅ Ensures horizontal grid stays
            gridcolor="rgba(255, 255, 255, 0.3)",  # ✅ Same color as x-axis
            linecolor="white",
            tickfont=dict(color="white")
        ),
        legend=dict(font=dict(color="white"))
    )


    st.plotly_chart(fig, use_container_width=True)

# 🌍 Global View Title - Reduce spacing even further
st.markdown("<h3 style='margin-bottom: -10px;'>🌍 Global View: Select Variable to Display on the Map</h3>", unsafe_allow_html=True)

# ✅ Move dropdown back to the left & ensure it works
selected_map_var = st.selectbox("Choose a variable to visualize", map_variables, key="map_variable")

# ✅ Wrap dropdown & map inside a container to fix spacing & layout
map_container = st.container()

with map_container:
    # ✅ Further reduce space between dropdown and map
    st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)

    # ✅ Filtered Data for the Map (Only Countries Being Displayed)
    map_df = df_filtered[["Country", "Continent", selected_map_var]].copy()

    # 🛠 Ensure numerical data and replace NaNs with a neutral color
    map_df[selected_map_var] = pd.to_numeric(map_df[selected_map_var], errors="coerce")
    map_df[selected_map_var] = map_df[selected_map_var].fillna(0)  # Missing values = neutral

    # 📌 Create the Choropleth Map
    fig_map = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color=selected_map_var,
        title=f"{selected_map_var} by Country",
        color_continuous_scale="RdYlGn",  # ✅ Red (low) → Yellow (medium) → Green (high)
        labels={selected_map_var: "Score"},  # ✅ Simplify legend
        template="plotly_dark"
    )

    # 📌 Ensure toolbar is inside the map
    fig_map.update_layout(
    geo=dict(
        showcoastlines=True,
        showland=True,
        landcolor="black"
    ),
    title=dict(
        font=dict(color="white"),
        x=0.5,
        xanchor="center"
    ),
    coloraxis_colorbar=dict(
        title=None,
        orientation="h",
        x=0.5,
        y=-0.25,  # ✅ Move the label up slightly (was -0.3)
        xanchor="center",
        yanchor="bottom"
    ),
    margin=dict(t=10, b=0, l=0, r=0),  # ✅ Reduce margins
)

# ✅ Ensure zoom/pan functions are inside the dark area of the map
st.plotly_chart(fig_map, use_container_width=True, config={
    "displayModeBar": True, 
    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d"],  
    "displaylogo": False
})











