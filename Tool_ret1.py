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
    
    # Download the file content from GitHub
    response = requests.get(url)
    response.raise_for_status()  # Ensure we stop on bad responses (e.g., 404)

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

# Categorize each variable into percentiles (quintiles)
def categorize_percentiles(df, variables):
    for var in variables:
        if var in df.columns:
            if var == "Pollution":
                # Pollution is inverted (higher value is worse)
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='min', ascending=False, na_option='bottom'),
                    5, labels=[5, 4, 3, 2, 1],
                    duplicates="drop"
                )
            elif var == "English Proficiency":
                # English Proficiency is already categorized
                english_mapping = {
                    "Very high proficiency": 1,
                    "High proficiency": 2,
                    "Moderate proficiency": 3,
                    "Low proficiency": 4,
                    "Very low proficiency": 5
                }
                df[f"{var}_Category"] = df[var].map(english_mapping)
            elif var in ["Openness", "Natural Scenery"]:
                # Lower rank is better
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='min', ascending=True, na_option='bottom'),
                    5, labels=[1, 2, 3, 4, 5],
                    duplicates="drop"
                )
            elif var == "Natural Disaster":
                # Higher values mean more disasters, so invert ranking
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='min', ascending=False, na_option='bottom'),
                    5, labels=[5, 4, 3, 2, 1],
                    duplicates="drop"
                )
            else:
                # Standard ranking for remaining variables (higher is better)
                df[f"{var}_Category"] = pd.qcut(
                    df[var].rank(method='first', ascending=True, na_option='bottom'),
                    5, labels=[5, 4, 3, 2, 1],
                    duplicates="drop"
                )
    return df



data = categorize_percentiles(data, data.columns[2:])  # Use `data`, not `df`



# DEBUG: Check if category columns exist
categorized_vars = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate",
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]

missing_categories = [f"{var}_Category" for var in categorized_vars if f"{var}_Category" not in data.columns]



# Title for the Tool
st.title("Best Countries for Early Retirement: Where to Retire Abroad?")

# Instructions Section
st.write("### Instructions for Using the Tool")
instructions = """
- This tool helps users identify the **best countries for retirement** abroad.  
- Use the **left panel** to **select variables** that matter to you (e.g., uncheck Pollution if it’s not a factor).  
- **Adjust sliders** to filter out low-performing countries for a given variable (e.g., setting Safety from 5 to 4 removes the least safe countries).  
- The tool calculates a **Retirement Suitability Score** as the average of the selected factors.  
- The **figure plots this score** against **cost of living (COL)** to highlight potential destinations.  
- Use the **zoom tool** (top-right of the figure) to explore specific countries.  
- Click on **legend continents** to hide/unhide regions.  
- **Example**: Setting strict criteria for Safety (2), Healthcare (2), Political Stability (4), Pollution (3), and Climate (3) results in 6 qualifying countries. Spain, Portugal, and Japan emerge as good candidates with a relatively low COL.  
- The **map below** shows how the Retirement Suitability factors are distributed geographically.  
- **Data is from Numbeo (2025)**, except for Political Stability, which is based on the **World Bank's Governance Indicators (2023)**.  
- The tool does not account for **Capital Gains Tax (CGT)**, but users should consider it (**[see here](https://taxsummaries.pwc.com/quick-charts/capital-gains-tax-cgt-rates)**).  
"""
st.write(instructions)


# Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
selected_vars = []
sliders = {}
variables = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate", 
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]
for label in variables:
    if st.sidebar.checkbox(label, value=True):
        sliders[label] = st.sidebar.slider(f"{label}", 1, 5, 5, format=None)
        selected_vars.append(label)

if selected_vars:
    # Start with full dataset
    df_filtered = data.copy()

    # Apply slider filters **before creating df_selected**
    for var in selected_vars:
        max_category = sliders[var]  
        category_col = f"{var}_Category"
    
        if category_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[category_col].astype(float) <= max_category]  

    # Now create df_selected from the already filtered data
    # Ensure selected variables exist in df_filtered
    # Ensure selected variables exist before filtering
    available_vars = [var for var in selected_vars if f"{var}_Category" in df_filtered.columns]
    
    # If no valid variables exist, stop execution
    if not available_vars:
        st.error("No valid variables selected. Please check the dataset or adjust your selections.")
        st.stop()
    
    df_selected = df_filtered[['Country', 'Cost of Living', 'Continent'] + [var for var in available_vars]].copy()


    
    # If no valid variables exist, stop execution
    if not available_vars:
        st.warning("No valid variables selected. Please check the dataset or adjust your selections.")
        st.stop()
    
    df_selected = df_filtered[['Country', 'Cost of Living', 'Continent'] + available_vars].copy()

    
    # If no valid variables exist, stop execution
    if not available_vars:
        st.stop()

    df_selected['Valid_Var_Count'] = df_selected[selected_vars].count(axis=1)
    df_selected['Retirement Suitability'] = df_selected[selected_vars].mean(axis=1)
    
    # ✅ Preserve selected continents across updates
    if "continent_selection" not in st.session_state:
        st.session_state["continent_selection"] = df_selected["Continent"].unique().tolist()  # Store initial selection
    
    # ✅ Multi-select continent filter (persists after slider changes)
    selected_continents = st.multiselect(
        "Select Continents to Display", 
        options=df_selected["Continent"].unique().tolist(), 
        default=st.session_state["continent_selection"]
    )
    
    # ✅ Apply the filter so only selected continents are shown
    df_selected = df_selected[df_selected["Continent"].isin(selected_continents)]
    
    # ✅ Store the selection so it persists across updates
    st.session_state["continent_selection"] = selected_continents

    df_selected['Retirement Suitability'] = df_selected[selected_vars].mean(axis=1)

    # Scatter Plot
    # Ensure we do not modify the original DataFrame
    df_selected = df_selected.copy()

    # If Pollution is in selected variables, create the inverse for hover display
    if "Pollution" in selected_vars:
        df_selected["Pollution_Hover"] = 100 - df_selected["Pollution"]  # Invert Pollution values

    # Update hover data mapping
    # Ensure hover text is formatted correctly
    hover_data_adjusted = {var: ':.2f' for var in selected_vars}
    hover_data_adjusted["Valid_Var_Count"] = True  
    
    # Special case: Adjust how Natural Disaster is displayed in the tooltip
    if "Natural Disaster" in selected_vars:
        hover_data_adjusted["Natural Disaster"] = ':.2f'


    # Ensure missing values are displayed as "NA"
    df_selected = df_selected.fillna("NA")  


    fig_scatter = px.scatter(
        df_selected, 
        x="Retirement Suitability", 
        y="Col_2025", 
        text="Country", 
        color=df_selected['Continent'],
        title="Retirement Suitability vs Cost of Living", 
        labels={
            "Col_2025": "Cost of Living (0 - 100)", 
            "Retirement Suitability": "Retirement Suitability (0 - 100)",
            "Pollution_Hover": "Pollution"
        },
        template="plotly_dark", 
        category_orders={"Continent": ["America", "Europe", "Asia", "Africa", "Oceania"]},
        hover_data=hover_data_adjusted
    )

    # ✅ Add checkbox below the scatter plot
    complete_data_only = st.checkbox("Show only countries with complete data", value=False)

    # ✅ Apply checkbox filter after defining it
    if complete_data_only:
        df_selected = df_selected[df_selected['Valid_Var_Count'] == len(selected_vars)]



    # ✅ Update scatter plot after filtering
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
        hover_data=hover_data_adjusted
    )

    fig_scatter.update_traces(marker=dict(size=10), textposition='top center')

    fig_scatter.update_layout(
        title=dict(text="Retirement Suitability vs Cost of Living", font=dict(color='white', size=24), x=0.5, xanchor="center"),
        xaxis=dict(linecolor='white', tickfont=dict(color='white'), showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1),
        yaxis=dict(linecolor='white', tickfont=dict(color='white'), showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1),
        legend=dict(font=dict(color="white")),
        paper_bgcolor='black', plot_bgcolor='black'
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
    


    # Map Visualization
    # Map Visualization
    st.write("### Understand the spatial distribution of the variables that make up the Retirement Suitability")
    selected_map_var = st.selectbox("", selected_vars)

    # Ensure selected variable is numeric and handle missing values
    df_selected[selected_map_var] = pd.to_numeric(df_selected[selected_map_var], errors='coerce')
    df_selected[selected_map_var].fillna(df_selected[selected_map_var].median(), inplace=True)
    
    # Generate the choropleth map
    fig_map = px.choropleth(
        df_selected, 
        locations="Country", 
        locationmode="country names", 
        color=selected_map_var, 
        color_continuous_scale="RdYlGn_r" if selected_map_var == "Natural Disaster" else "RdYlGn",
        title=f"{selected_map_var} by Country",
        labels={selected_map_var: selected_map_var}  # Ensures label is correct
    )
    
    # Format the layout for a consistent display
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar=dict(title=selected_map_var)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)






