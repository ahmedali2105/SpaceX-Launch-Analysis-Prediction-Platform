import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
import joblib
import os
import sys

# Add the parent directory to the system path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from data_fetcher and model_trainer
from data_fetcher import load_and_preprocess_data
from model_trainer import load_model_and_features, train_model

# --- Configuration ---
st.set_page_config(
    page_title="SpaceX Launch Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Variables / Caching ---
@st.cache_data
def get_data():
    """Caches the data loading and preprocessing."""
    df_display, df_model = load_and_preprocess_data()
    return df_display, df_model

@st.cache_resource
def get_model():
    """Caches the model loading and training."""
    model, feature_columns = load_model_and_features()
    if model is None or feature_columns is None:
        st.warning("Model not found or failed to load. Attempting to train a new model...")
        df_display, df_model = get_data() # Ensure data is available for training
        if not df_model.empty:
            model, feature_columns = train_model(df_model)
            if model is None:
                st.error("Failed to train a new model. Prediction functionality will be limited.")
        else:
            st.error("No data available to train the model. Prediction functionality will be disabled.")
    return model, feature_columns

df_display, df_model = get_data()
model, feature_columns = get_model()

# --- Helper Functions ---
def make_prediction(input_data, model, feature_columns):
    """
    Makes a prediction using the trained model.
    Ensures input data has the same features as the training data.
    """
    if model is None or feature_columns is None:
        st.error("Prediction model is not available.")
        return None

    # Create a DataFrame from input_data
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features, ensuring all original columns are present
    # This is crucial for consistent prediction
    # Create a dummy DataFrame with all possible columns from training data
    # and then fill with input values.
    
    # Identify categorical columns that were one-hot encoded during training
    categorical_cols_trained = [col.replace('rocket_name_', '').replace('launchpad_name_', '')
                                for col in feature_columns if 'rocket_name_' in col or 'launchpad_name_' in col]
    
    # Create a template DataFrame with all expected columns, initialized to 0
    template_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill in the numerical values from input_df
    for col in ['payload_mass_kg', 'core_reused', 'year']:
        if col in input_df.columns:
            template_df[col] = input_df[col].values[0]

    # Handle one-hot encoding for 'rocket_name'
    if 'rocket_name' in input_df.columns:
        rocket_val = input_df['rocket_name'].values[0]
        col_name = f'rocket_name_{rocket_val}'
        if col_name in template_df.columns:
            template_df[col_name] = 1

    # Handle one-hot encoding for 'launchpad_name'
    if 'launchpad_name' in input_df.columns:
        launchpad_val = input_df['launchpad_name'].values[0]
        col_name = f'launchpad_name_{launchpad_val}'
        if col_name in template_df.columns:
            template_df[col_name] = 1

    try:
        prediction_proba = model.predict_proba(template_df)[0][1] # Probability of success (class 1)
        return prediction_proba
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- App Title and Description ---
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("""
    This platform allows you to explore historical SpaceX launch data and predict the success
    of future launches using a machine learning model.
""")

if df_display.empty:
    st.error("Failed to load or preprocess SpaceX launch data. Please check your internet connection or try again later.")
else:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    all_years = sorted(df_display['year'].unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", all_years, default=all_years)

    all_launchpads = sorted(df_display['launchpad_name'].unique())
    selected_launchpads = st.sidebar.multiselect("Select Launch Site(s)", all_launchpads, default=all_launchpads)

    filtered_df = df_display[
        (df_display['year'].isin(selected_years)) &
        (df_display['launchpad_name'].isin(selected_launchpads))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    st.sidebar.markdown("---")
    st.sidebar.header("Prediction Parameters")
    st.sidebar.markdown("Use these inputs for the prediction tool below.")

    # --- Main Content ---

    st.header("Historical Launch Data")
    st.write("Explore past SpaceX launches with filters.")

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.dataframe(filtered_df[['flight_number', 'name', 'date_utc', 'rocket_name', 'launchpad_name', 'payload_mass_kg', 'launch_success', 'core_reused']].sort_values('date_utc', ascending=False))

        # --- EDA Visualizations ---
        st.header("Exploratory Data Analysis (EDA)")

        # Launch Success Rate by Rocket Type
        st.subheader("Launch Success Rate by Rocket Type")
        success_by_rocket = filtered_df.groupby('rocket_name')['launch_success'].mean().reset_index()
        fig_rocket = px.bar(
            success_by_rocket,
            x='rocket_name',
            y='launch_success',
            title='Launch Success Rate by Rocket Type',
            labels={'launch_success': 'Success Rate', 'rocket_name': 'Rocket Type'},
            color='launch_success',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_rocket, use_container_width=True)

        # Launch Success Rate by Launch Site
        st.subheader("Launch Success Rate by Launch Site")
        success_by_launchpad = filtered_df.groupby('launchpad_name')['launch_success'].mean().reset_index()
        fig_launchpad = px.bar(
            success_by_launchpad,
            x='launchpad_name',
            y='launch_success',
            title='Launch Success Rate by Launch Site',
            labels={'launch_success': 'Success Rate', 'launchpad_name': 'Launch Site'},
            color='launch_success',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_launchpad, use_container_width=True)

        # Payload Mass vs. Launch Success
        st.subheader("Payload Mass vs. Launch Success")
        fig_payload = px.scatter(
            filtered_df,
            x='payload_mass_kg',
            y='flight_number', # Using flight number as y-axis to show distribution
            color='launch_success',
            title='Payload Mass vs. Launch Success',
            labels={'payload_mass_kg': 'Payload Mass (kg)', 'flight_number': 'Flight Number', 'launch_success': 'Launch Success (0=Fail, 1=Success)'},
            hover_data=['name', 'rocket_name', 'launchpad_name'],
            color_discrete_map={0: 'red', 1: 'green'}
        )
        st.plotly_chart(fig_payload, use_container_width=True)


        # --- Geospatial Map ---
        st.header("Geospatial Map of Launch Sites")
        st.write("Visualize SpaceX launch sites and outcomes on a map.")

        # Ensure latitude and longitude are not NaN for mapping
        map_df = filtered_df.dropna(subset=['latitude', 'longitude']).copy()

        if not map_df.empty:
            # Create a base map centered around the average launch site location
            m = folium.Map(location=[map_df['latitude'].mean(), map_df['longitude'].mean()], zoom_start=4)

            # Add markers for each launch
            for idx, row in map_df.iterrows():
                color = 'green' if row['launch_success'] == 1 else 'red'
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    tooltip=f"Launch: {row['name']}<br>Success: {'Yes' if row['launch_success'] == 1 else 'No'}<br>Site: {row['launchpad_name']}"
                ).add_to(m)
            
            # Display the map
            folium_static(m, width=700, height=500)
        else:
            st.warning("No launch site data with coordinates available for mapping with current filters.")


        # --- Predictive Tool ---
        st.header("Predict Future Launch Success")
        st.write("Input parameters to get a prediction for a future SpaceX launch.")

        if model is None or feature_columns is None:
            st.error("Prediction model is not loaded or trained. Please ensure `model_trainer.py` has been run or check for errors during model loading.")
        else:
            # Input fields for prediction
            col1, col2, col3 = st.columns(3)

            with col1:
                predict_payload_mass = st.number_input("Payload Mass (kg)", min_value=0, value=5000, step=100)
            with col2:
                predict_rocket_name = st.selectbox("Rocket Type", options=df_display['rocket_name'].unique())
            with col3:
                predict_launchpad_name = st.selectbox("Launch Site", options=df_display['launchpad_name'].unique())

            predict_core_reused = st.checkbox("Core Reused?", value=True)
            predict_year = st.number_input("Launch Year", min_value=2020, value=2025, step=1)

            predict_button = st.button("Predict Launch Success Probability")

            if predict_button:
                input_data = {
                    'payload_mass_kg': predict_payload_mass,
                    'core_reused': 1 if predict_core_reused else 0,
                    'year': predict_year,
                    'rocket_name': predict_rocket_name,
                    'launchpad_name': predict_launchpad_name
                }
                
                # Ensure the input data has the same structure as the training data
                # This requires creating a DataFrame with all possible one-hot encoded columns
                # and setting the relevant ones to 1.

                # Create a DataFrame with all feature_columns and set to 0 initially
                input_df_processed = pd.DataFrame(0, index=[0], columns=feature_columns)

                # Populate numerical features
                input_df_processed['payload_mass_kg'] = input_data['payload_mass_kg']
                input_df_processed['core_reused'] = input_data['core_reused']
                input_df_processed['year'] = input_data['year']

                # Populate one-hot encoded features
                # Rocket name
                rocket_col = f'rocket_name_{input_data["rocket_name"]}'
                if rocket_col in input_df_processed.columns:
                    input_df_processed[rocket_col] = 1
                else:
                    st.warning(f"Rocket type '{input_data['rocket_name']}' not seen during training. This might affect prediction accuracy.")

                # Launchpad name
                launchpad_col = f'launchpad_name_{input_data["launchpad_name"]}'
                if launchpad_col in input_df_processed.columns:
                    input_df_processed[launchpad_col] = 1
                else:
                    st.warning(f"Launch site '{input_data['launchpad_name']}' not seen during training. This might affect prediction accuracy.")

                # Make prediction
                try:
                    prediction_proba = model.predict_proba(input_df_processed)[0][1]
                    st.success(f"Predicted Launch Success Probability: **{prediction_proba:.2f}**")
                    if prediction_proba >= 0.5:
                        st.balloons()
                        st.write("This launch has a high probability of success!")
                    else:
                        st.write("This launch has a lower probability of success.")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.info("Please ensure all input parameters are valid and the model is correctly loaded.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Syed Ahmed Ali | Powered by Machine Learning")

