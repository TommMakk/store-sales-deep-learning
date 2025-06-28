from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import typer
from loguru import logger
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
from store_sales_DL.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()

def plot_sales_over_time(df, date_col='date', sales_col='sales', legend_col='History or predicted', title='Sales Over Time'):
    fig = px.line(
        df,
        x=date_col,
        y=sales_col,
        color=legend_col,
        title=title,
        markers=True,
        template="plotly_white"
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        title_x=0.5,
        font=dict(size=16),
        legend_title_text="History or predicted",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            valign='bottom',
            title_text='History or predicted',
            itemwidth=30,
            itemsizing='constant'
        ),
        margin=dict(b=60)
    )
    return fig

def family_tile_selector(df, family_col='family', n_cols=4):
    """
    Streamlit widget for selecting families using clickable tiles arranged in columns.
    'GROCERY I' is ticked by default. Returns a list of selected families.
    """
    families = sorted(df[family_col].unique())
    select_all_key = "family_select_all"
    selected_families_key = "selected_families"

    # Initialize session state: 'GROCERY I' selected by default
    if select_all_key not in st.session_state:
        st.session_state[select_all_key] = False
    if selected_families_key not in st.session_state:
        st.session_state[selected_families_key] = ['GROCERY I'] if 'GROCERY I' in families else []

    # Select All checkbox
    select_all = st.checkbox("Select All Families", value=st.session_state[select_all_key])
    if select_all:
        st.session_state[selected_families_key] = families.copy()
        st.session_state[select_all_key] = True
    else:
        # If unticked, clear all selections
        if st.session_state[select_all_key]:
            st.session_state[selected_families_key] = ['GROCERY I'] if 'GROCERY I' in families else []
        st.session_state[select_all_key] = False

    # Arrange family tiles in 4 columns, left-to-right
    n_rows = (len(families) + n_cols - 1) // n_cols
    cols = st.columns(n_cols)

    for row in range(n_rows):
        for col in range(n_cols):
            idx = col + row * n_cols  # <-- changed here
            if idx < len(families):
                family = families[idx]
                checked = family in st.session_state[selected_families_key]
                if cols[col].checkbox(family, value=checked, key=f"family_{family}"):
                    if family not in st.session_state[selected_families_key]:
                        st.session_state[selected_families_key].append(family)
                else:
                    if family in st.session_state[selected_families_key]:
                        st.session_state[selected_families_key].remove(family)

    # Keep 'Select All' checked if all families are selected, otherwise keep it unchecked
    if set(st.session_state[selected_families_key]) == set(families):
        st.session_state[select_all_key] = True
    elif len(st.session_state[selected_families_key]) == 0:
        st.session_state[select_all_key] = False

    return st.session_state[selected_families_key]

def date_filter_widget(df, date_col='date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    years = sorted(df[date_col].dt.year.unique())
    months = sorted(df[date_col].dt.month.unique())

    col1, col2, col3 = st.columns([2, 2, 6])

    with col1:
        selected_year = st.selectbox("Year", options=["All"] + years, index=0)
    with col2:
        selected_month = st.selectbox("Month", options=["All"] + months, index=0)
    with col3:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range = st.date_input(
            "Or select a date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD"
        )

    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df[date_col].dt.year == selected_year]
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df[date_col].dt.month == selected_month]
    if selected_year == "All" and selected_month == "All":
        # Handle empty date_range
        if not date_range or len(date_range) != 2:
            start_date, end_date = min_date, max_date
        else:
            start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df[date_col] >= pd.to_datetime(start_date)) &
                                  (filtered_df[date_col] <= pd.to_datetime(end_date))]
    return filtered_df

def ecuador_map(cities_df, provinces_geojson):
    # Create a DataFrame for provinces with a constant column for legend
    province_names = [feature['properties']['DPA_DESPRO'] for feature in provinces_geojson['features']]
    provinces_df = pd.DataFrame({'DPA_DESPRO': province_names, 'Ecuador': 'state'})

    # Plot provinces as boundaries with legend "Ecuador" and dark map style
    fig = px.choropleth_mapbox(
        provinces_df,
        geojson=provinces_geojson,
        locations='DPA_DESPRO',
        featureidkey="properties.DPA_DESPRO",
        color='Ecuador',
        color_discrete_map={'state': '#b3cde3'},
        mapbox_style="carto-darkmatter",  # <-- Dark theme here
        center={"lat": -1.8312, "lon": -78.1834},
        zoom=5.5,
        opacity=0.3
    )

    # Add city points
    fig.add_scattermapbox(
        lat=cities_df['lat'],
        lon=cities_df['lon'],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=cities_df['city'],
        textposition="top right",
        name="city"
    )

    fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    height=610,
    legend=dict(
        orientation="v",
        x=0.01,         # a little inside from the left
        y=0.99,         # a little inside from the top
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(0,0,0,0.5)',  # semi-transparent background for readability
        font=dict(color='white'),   # white text for dark maps
        title_text='Ecuador'
    )
)
    return fig

#@app.command()
def main(
   # Define the input paths
    dashboard_dataset_path = PROCESSED_DATA_DIR / "dashboard_data.csv",
    stores_with_coords_path = REPORTS_DIR / "stores_fully_updated_with_coordinates.csv",
    ecuador_geojson_path = REPORTS_DIR / "ecuador.geojson"

):

    logger.info("Starting to form streamlit app...")

    # Load your data
    df = pd.read_csv(dashboard_dataset_path)
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is datetime

    #st.title("Store Sales Dashboard")
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>Ecuador store sales dashboard</h1>", unsafe_allow_html=True)
    # Create two main columns: left (filters), right (outputs)
    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.container():
            st.subheader("Date related filters")

            # Prepare options
            all_years = sorted(df['date'].dt.year.unique())
            default_years = [2017] if 2017 in all_years else []
            all_months = sorted(df['date'].dt.month.unique())
            month_names = {i: pd.to_datetime(str(i), format='%m').strftime('%B') for i in all_months}
            month_options = [month_names[m] for m in all_months]
            all_weeks = sorted(df['date'].dt.isocalendar().week.unique())
            all_weekdays = list(range(0, 7))
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            # Stack year, month, week, weekday horizontally
            col_year, col_month, col_week, col_weekday = st.columns(4)
            with col_year:
                selected_years = st.multiselect("Year(s)", options=all_years, default=default_years, key="year_multiselect")
            with col_month:
                selected_month_names = st.multiselect("Month(s)", options=month_options, default=[], key="month_multiselect")
                selected_months = [m for m, name in month_names.items() if name in selected_month_names]
            with col_week:
                selected_weeks = st.multiselect("Week(s)", options=all_weeks, default=[], key="week_multiselect")
            with col_weekday:
                selected_weekdays = st.multiselect("Weekday(s)", options=weekday_names, default=[], key="weekday_multiselect")
                selected_weekday_nums = [i for i, name in enumerate(weekday_names) if name in selected_weekdays]

            # Date range filter (can be placed below)
            min_date = df['date'].min()
            max_date = df['date'].max()
            date_range = st.date_input(
                "Or select a date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                format="YYYY-MM-DD",
                key="date_range"
            )

            # Apply filters
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['date'].dt.year.isin(selected_years)]
            if selected_months:
                filtered_df = filtered_df[filtered_df['date'].dt.month.isin(selected_months)]
            if selected_weeks:
                filtered_df = filtered_df[filtered_df['date'].dt.isocalendar().week.isin(selected_weeks)]
            if selected_weekday_nums:
                filtered_df = filtered_df[filtered_df['date'].dt.weekday.isin(selected_weekday_nums)]
            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(start_date)) &
                                        (filtered_df['date'] <= pd.to_datetime(end_date))]
        with st.container():
            st.subheader("Product family filter")
            selected_family = family_tile_selector(filtered_df, family_col='family', n_cols=3)
            filtered_df = filtered_df[filtered_df['family'].isin(selected_family)]

        with st.container():
            st.subheader("Store related filters")

            # State multiselect with "All" and default "Pichincha"
            all_states = df['state'].dropna().unique().tolist()
            state_options = ["All"] + all_states
            default_states = ["Pichincha"] if "Pichincha" in all_states else []
            selected_states = st.multiselect(
                "Select State(s)", options=state_options, default=default_states, key="state_multiselect"
            )
            if "All" in selected_states or not selected_states:
                selected_states = all_states

            # City multiselect with "All" and default "Cayambe"
            all_cities = df[df['state'].isin(selected_states)]['city'].dropna().unique().tolist()
            city_options = ["All"] + all_cities
            default_cities = ["Cayambe"] if "Cayambe" in all_cities else []
            selected_cities = st.multiselect(
                "Select City(s)", options=city_options, default=default_cities, key="city_multiselect"
            )
            if "All" in selected_cities or not selected_cities:
                selected_cities = all_cities

            # Store multiselect with "All" and default 11
            all_stores = df[
                (df['state'].isin(selected_states)) &
                (df['city'].isin(selected_cities))
            ]['store_nbr'].unique().tolist()
            store_options = ["All"] + all_stores
            default_stores = [11] if 11 in all_stores else []
            selected_stores = st.multiselect(
                "Select Store(s)", options=store_options, default=default_stores, key="store_multiselect"
            )
            if "All" in selected_stores or not selected_stores:
                selected_stores = all_stores

            # Apply hierarchical filter
            filtered_df = filtered_df[
                (filtered_df['state'].isin(selected_states)) &
                (filtered_df['city'].isin(selected_cities)) &
                (filtered_df['store_nbr'].isin(selected_stores))
            ]


    # Load data to map
    cities_df = pd.read_csv(stores_with_coords_path)
    with open(ecuador_geojson_path, 'r') as f:
        provinces_geojson = json.load(f)

    # Prepare table data
    table_df = filtered_df.copy()
    table_df['date'] = table_df['date'].dt.date
    table_df['sales'] = table_df['sales'].round(0).astype(int)
    table_df = table_df.sort_values('date', ascending=False)
    table_df = table_df[['date', 'store_nbr', 'family', 'sales', 'History or predicted']]

    # Define colors matching your "sales over time" chart
    history_color = "#636EFA"      # Plotly blue
    predicted_color = "#EF553B"    # Plotly red

    def color_history_predicted(val):
        if val == "History":
            return f"color: {history_color};"
        elif val == "Predicted":
            return f"color: {predicted_color};"
        return ""

    styled_table = table_df.style.applymap(color_history_predicted, subset=['History or predicted'])

    # Right column: Outputs
    with col_right:
        with st.container():
            if not filtered_df.empty:
                fig = plot_sales_over_time(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")
        with st.container():
            col_table, col_map = st.columns([1, 2])
            with col_table:
                st.dataframe(styled_table, height=610, hide_index=True)
            with col_map:
                fig_map = ecuador_map(cities_df, provinces_geojson)
                st.plotly_chart(fig_map, use_container_width=True)
    logger.success("Plot generation complete.")
    #streamlit run store_sales_DL/dashboard.py


if __name__ == "__main__":
    main()




