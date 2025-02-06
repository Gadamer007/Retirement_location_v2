import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from io import BytesIO

st.markdown("""
    <style>
    /* Increase Sidebar Width */
    [data-testid="stSidebar"] {
        min-width: 360px !important;  /* Wider sidebar */
        max-width: 360px !important;
    }

    /* Reduce font size for checkboxes (variable labels) */
    [data-testid="stSidebar"] label {
        font-size: 11px !important;  /* Smaller text */
        font-weight: 500 !important; /* Keep labels readable */
        margin-bottom: -10px !important; /* Reduce space between checkbox and slider */
        white-space: nowrap !important; /* Prevent text from wrapping */
    }

    /* Reduce space between checkbox and slider */
    [data-testid="stSidebar"] .st-bb {
        margin-bottom: -15px !important; /* Brings slider closer to the label */
    }

    /* Reduce spacing between sliders to ensure proper alignment */
    [data-testid="stSidebar"] .st-br {
        margin-bottom: 10px !important; /* More spacing between sliders */
    }

    /* Ensure checkboxes and sliders have more horizontal space */
    [data-testid="stSidebarContent"] {
        padding-right: 15px !important; /* Adds some padding to prevent text cutoff */
    }
    </style>
    """, unsafe_allow_html=True)





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
st.sidebar.subheader("Select Variables for Retirement Suitability")
st.sidebar.markdown("<small>Move slider to the left to drop worst performing countries. For example, moving slider from 5 to 4 drops the bottom 20% performing countries</small>", unsafe_allow_html=True)
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
def normalize_and_categorize(df, variables):
    for var in variables:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')  # Ensure numeric
            df[var] = df[var].replace([np.inf, -np.inf], np.nan)  # Remove infinite values but keep NaN
            
            # Normalize all variables (except missing values remain as NaN)
            min_val = df[var].min()
            max_val = df[var].max()
            
            # Special case for Openness & Natural Scenery (ranking transformation)
            if var in ["Openness", "Natural Scenery"]:
                min_rank = df[var].min()
                max_rank = df[var].max()
                
                # Convert rank to 0-100 scale (1 = 100, worst = 0)
                df[var] = ((max_rank - df[var]) / (max_rank - min_rank)) * 100  
            
                # Ensure transformed values are used for plotting & averaging
                df[var] = df[var].round(2)  # Keep 2 decimal places for clarity
            
            else:
                if min_val != max_val:  # Avoid division by zero
                    df[var] = (df[var] - min_val) / (max_val - min_val) * 100
            
            # Reverse Pollution normally (low is good, high is bad)
            # Reverse Pollution so that lower values (clean air) = better
            if var == "Pollution":
                min_pollution = df[var].min()
                max_pollution = df[var].max()
                
                df[var] = ((max_pollution - df[var]) / (max_pollution - min_pollution)) * 100  # Cleaner air = higher score

            
            # Reverse Natural Disaster to ensure "safer = better"
            if var == "Natural Disaster":
                df[var] = df[var].rank(pct=True) * 100  # Convert rank to 0-100 percentile-based scaling (safest = 100)
                
                threshold = np.percentile(df[var].dropna(), (6 - max_category) * 20)  # Get top 20%, 40%, etc.
                df_filtered = df_filtered[df_filtered[var] >= threshold]  # Keep top safest countries

            
            try:
                # Variables where HIGH values are BETTER (rank from high to low)
                direct_scale_vars = ["Safety", "Healthcare", "Political Stability", "Climate", "Openness", "Natural Scenery"]
                
                # Variables where LOW values are BETTER (rank from low to high)
                inverse_scale_vars = ["Pollution", "Natural Disaster"]
            
                if var in inverse_scale_vars:
                    df[f"{var}_Category"] = pd.qcut(
                        df[var].rank(method='min', ascending=False, na_option='bottom'),  # Lower values ranked higher
                        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
                    )
                else:
                    df[f"{var}_Category"] = pd.qcut(
                        df[var].rank(method='min', ascending=True, na_option='bottom'),  # Higher values ranked higher
                        q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"  # Reverse labels: 1 = best, 5 = worst
                    )
            except ValueError:
                df[f"{var}_Category"] = pd.cut(
                    df[var].rank(method='min', ascending=(var in inverse_scale_vars), na_option='bottom'),
                    bins=5, labels=[1, 2, 3, 4, 5] if var in inverse_scale_vars else [5, 4, 3, 2, 1],
                    include_lowest=True
                )

            
            # Define reversed scales (lower values = better)
            inverse_vars = ["Pollution", "Natural Disaster"]
            
            if var in inverse_vars:
                df[var] = 100 - df[var]  # Invert values (higher becomes worse)

            try:
                # Variables where HIGH values are BETTER (rank from high to low)
                direct_scale_vars = ["Safety", "Healthcare", "Political Stability", "Climate", "Openness", "Natural Scenery"]
                
                # Variables where LOW values are BETTER (rank from low to high)
                inverse_scale_vars = ["Pollution", "Natural Disaster"]
            
                if var in inverse_scale_vars:
                    df[f"{var}_Category"] = pd.qcut(
                        df[var].rank(method='min', ascending=False, na_option='bottom'),  # Lower values ranked higher
                        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
                    )
                else:
                    df[f"{var}_Category"] = pd.qcut(
                        df[var].rank(method='min', ascending=True, na_option='bottom'),  # Higher values ranked higher
                        q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"  # Reverse labels: 1 = best, 5 = worst
                    )
            except ValueError:
                df[f"{var}_Category"] = pd.cut(
                    df[var].rank(method='min', ascending=(var in inverse_scale_vars), na_option='bottom'),
                    bins=5, labels=[1, 2, 3, 4, 5] if var in inverse_scale_vars else [5, 4, 3, 2, 1],
                    include_lowest=True
                )

    return df



# Apply normalization and ranking
data = normalize_and_categorize(data, variables)  # Ensures Climate and other variables are correctly ranked

# Sidebar layout using columns for a more compact arrangement
cols = st.sidebar.columns(2)  # Two-column layout
i = 0
selected_vars = []  # Reset selected variables list
sliders = {}  # Reset sliders dictionary

for label in variables:
    category_col = f"{label}_Category"
    if category_col in data.columns:
        with cols[i % 2]:  # Each variable + slider stays together
            checked = st.checkbox(f"**{label}**", value=True, key=f"check_{label}")  # Keep checkboxes, now bold
            if checked:  # Only show slider if checkbox is selected
                st.markdown(f"<div style='height:2px'></div>", unsafe_allow_html=True)  # Small gap for alignment
                sliders[label] = st.slider(
                    "",  # Empty label for slider
                    1, 5, 5, 
                    key=f"slider_{label}"
                )
                selected_vars.append(label)  # Add to selected variables list
        i += 1  # Move to next column








if not selected_vars:
    st.error("⚠️ No valid variables selected. Please check the dataset or adjust your selections.")
    st.stop()

# Start with full dataset
# Start with full dataset
df_filtered = data.copy()

# Apply slider filters using actual 0-100 transformed values instead of category ranks
for var in selected_vars:
    max_category = sliders[var]  # Slider value (1-5)

    if var in ["Openness", "Natural Scenery"]:
        df[var] = df[var].rank(pct=True) * 100  # Convert rank to 0-100 percentile-based scaling
        
        threshold = np.percentile(df[var].dropna(), (6 - max_category) * 20)  # Get top 20%, 40%, etc.
        df_filtered = df_filtered[df_filtered[var] >= threshold]  # Keep countries above threshold

    else:
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
df_selected['Retirement Suitability'] = df_selected[real_value_vars].mean(axis=1, skipna=False).round(2)


# Add the selected variables' real values (0-100) to the dataframe
for var in real_value_vars:
    df_selected[var] = data[var]  # Add actual values, not categories

# Compute Retirement Suitability Score using actual values
if real_value_vars:
    df_selected['Retirement Suitability'] = df_selected[real_value_vars].mean(axis=1, skipna=True).round(2)
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

# Add a checkbox to filter out countries with more than 2 missing values
exclude_incomplete = st.checkbox("Exclude countries with incomplete data (more than 2 N/A)")

# Apply the filter if the checkbox is selected
if exclude_incomplete:
    df_selected = df_selected[df_selected.isna().sum(axis=1) <= 2]


# Scatter Plot
# Ensure correct hover data references to _Category columns
hover_data_adjusted = {f"{var}_Category": ':.2f' for var in selected_vars if f"{var}_Category" in df_selected.columns}

# Fix hover data: Show actual values, ensuring no duplicates
hover_data_adjusted = {
    "Continent": True,  # Ensure Continent appears first
    "Country": True,  # Then Country
    "Retirement Suitability": ':.2f'  # Round suitability score
}

# Add real values (0-100) for selected variables, ensuring no duplicates
for var in real_value_vars:
    if var in df_selected.columns:  # Ensure it's already in df_selected to prevent conflicts
        hover_data_adjusted = {var: (':.2f' if var in df_selected.columns else None) for var in real_value_vars}


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
    hover_data=hover_data_adjusted  # ✅ Fix applied here
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







