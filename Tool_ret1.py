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

# Define variables to categorize
categorized_vars = [
    "Safety", "Healthcare", "Political Stability", "Pollution", "Climate",
    "English Proficiency", "Openness", "Natural Scenery", "Natural Disaster"
]

# Ensure the dataset contains the required columns
missing_columns = [var for var in categorized_vars if var not in data.columns]
if missing_columns:
    st.error(f"⚠️ The dataset is missing these expected columns: {missing_columns}")
    st.stop()

# Sidebar Filters
st.sidebar.subheader("Select Variables for Retirement Suitability")
selected_vars = []
sliders = {}

for label in categorized_vars:
    if st.sidebar.checkbox(label, value=True):
        sliders[label] = st.sidebar.slider(f"{label}", 1, 5, 5)
        selected_vars.append(label)

if not selected_vars:
    st.error("No valid variables selected. Please select at least one variable.")
    st.stop()

# Start with full dataset
df_filtered = data.copy()

# Apply filters
for var in selected_vars:
    max_value = sliders[var]  
    if var in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[var] <= max_value]

# Check if filtering removed all data
if df_filtered.empty:
    st.warning("No countries match the selected criteria. Try adjusting the filters.")
    st.stop()

# Final selection
df_selected = df_filtered[['Country', 'Cost of Living', 'Continent'] + selected_vars].copy()
df_selected['Retirement Suitability'] = df_selected[selected_vars].mean(axis=1)

# Display the filtered data
st.write("### Filtered Data")
st.dataframe(df_selected)

# Plot the data
fig = px.scatter(
    df_selected,
    x="Retirement Suitability",
    y="Cost of Living",
    text="Country",
    color="Continent",
    title="Retirement Suitability vs Cost of Living"
)
st.plotly_chart(fig, use_container_width=True)







