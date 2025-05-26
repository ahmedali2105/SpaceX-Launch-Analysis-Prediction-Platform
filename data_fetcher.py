import requests
import pandas as pd
import numpy as np
import os

# Define the path for saving data
DATA_DIR = 'data'
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'spacex_launch_data.csv')

def fetch_spacex_data():
    """
    Fetches historical SpaceX launch data from the SpaceX-API.
    """
    url = "https://api.spacexdata.com/v4/launches/past"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching SpaceX data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocesses the raw SpaceX launch data.
    - Extracts relevant features.
    - Handles missing values.
    - Encodes categorical variables.
    - Creates a 'launch_success' target variable.
    """
    if not data:
        return pd.DataFrame()

    # Create a list to store extracted features
    processed_launches = []

    for launch in data:
        # Basic error handling for missing keys
        launch_id = launch.get('id')
        flight_number = launch.get('flight_number')
        name = launch.get('name')
        date_utc = launch.get('date_utc')
        success = launch.get('success')
        details = launch.get('details')
        
        # Rocket details
        rocket_id = launch.get('rocket') # This is an ID, needs to be resolved to name
        
        # Launchpad details
        launchpad_id = launch.get('launchpad') # This is an ID, needs to be resolved to name and coordinates
        
        # Payload details (simplistic for now, usually payloads is a list of IDs)
        # For simplicity, we'll just check if payloads exist and get a count or a representative mass later
        payloads = launch.get('payloads', [])
        
        # Attempt to get payload mass from the first payload, if available
        # This is a simplification; a real app would fetch payload details via their API
        payload_mass_kg = None
        if payloads:
            # This requires fetching payload details from another endpoint,
            # which is beyond the scope of a single function call here.
            # For now, we'll use a placeholder or assume average if we had a lookup.
            # Let's just use a dummy value or fill NaN later.
            pass # We'll fill this with a placeholder or imputation later

        # Cores details (for reusability)
        cores = launch.get('cores', [])
        core_reused = any(core.get('reused', False) for core in cores) if cores else False

        processed_launches.append({
            'id': launch_id,
            'flight_number': flight_number,
            'name': name,
            'date_utc': date_utc,
            'launch_success': 1 if success else 0, # Target variable
            'details': details,
            'rocket_id': rocket_id,
            'launchpad_id': launchpad_id,
            'core_reused': core_reused,
            # Placeholder for payload mass, will be imputed or handled later
            'payload_mass_kg': np.nan # Will be filled by fetching more data or imputation
        })

    df = pd.DataFrame(processed_launches)

    if df.empty:
        print("No data to preprocess.")
        return df

    # Convert date to datetime and extract year
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    df['year'] = df['date_utc'].dt.year

    # --- Fetching Rocket and Launchpad Names (Requires additional API calls) ---
    # For a robust solution, you'd fetch these from their respective endpoints:
    # https://api.spacexdata.com/v4/rockets/{rocket_id}
    # https://api.spacexdata.com/v4/launchpads/{launchpad_id}
    # For this example, we'll use a simplified mapping or leave as IDs if not critical for ML.
    # Let's create dummy mappings for demonstration.
    
    # Dummy rocket names
    rocket_mapping = {
        "5e9d0d95eda69973a809d1ec": "Falcon 9",
        "5e9d0d95eda69974db09d1ed": "Falcon Heavy",
        "5e9d0d95eda69955f709d1eb": "Starship" # Example, may not be in past launches
    }
    df['rocket_name'] = df['rocket_id'].map(rocket_mapping).fillna('Unknown Rocket')

    # Dummy launchpad names and coordinates
    launchpad_mapping = {
        "5e9e4501f509094ba4566f84": {"name": "Cape Canaveral SLC 40", "latitude": 28.561941, "longitude": -80.577357},
        "5e9e4502f509092b78566f87": {"name": "Kennedy Space Center LC 39A", "latitude": 28.608389, "longitude": -80.603957},
        "5e9e4502f5090995ed566f86": {"name": "Vandenberg AFB SLC 4E", "latitude": 34.632093, "longitude": -120.610829}
    }
    df['launchpad_name'] = df['launchpad_id'].map(lambda x: launchpad_mapping.get(x, {}).get('name', 'Unknown Launchpad'))
    df['latitude'] = df['launchpad_id'].map(lambda x: launchpad_mapping.get(x, {}).get('latitude', np.nan))
    df['longitude'] = df['launchpad_id'].map(lambda x: launchpad_mapping.get(x, {}).get('longitude', np.nan))

    # Impute missing payload_mass_kg with the mean
    # This is a very simple imputation. In a real scenario, you might use more sophisticated methods
    # or fetch actual payload data.
    if df['payload_mass_kg'].isnull().any():
        # Let's assign a random mass within a reasonable range for demonstration
        # as we don't have actual payload mass from the /launches endpoint directly.
        # A better approach would be to fetch data from /payloads endpoint.
        # For now, we'll use a fixed value or mean of a hypothetical range.
        df['payload_mass_kg'] = df['payload_mass_kg'].fillna(df['payload_mass_kg'].mean() if not df['payload_mass_kg'].isnull().all() else 10000) # Default to 10000 if all are NaN

    # Select features for the ML model
    # For simplicity, we'll use numerical and one-hot encoded categorical features
    features = ['payload_mass_kg', 'core_reused', 'year', 'rocket_name', 'launchpad_name']
    df_model = df[features + ['launch_success']].copy()

    # One-hot encode categorical features
    df_model = pd.get_dummies(df_model, columns=['rocket_name', 'launchpad_name'], drop_first=True)

    # Handle any remaining NaNs (e.g., from new launch sites/rockets not in mapping)
    df_model = df_model.fillna(0) # Fill with 0 for one-hot encoded columns

    # Ensure all columns are numeric for the model
    for col in df_model.columns:
        if df_model[col].dtype == 'bool':
            df_model[col] = df_model[col].astype(int)

    return df, df_model

def load_and_preprocess_data():
    """
    Loads preprocessed data if available, otherwise fetches and preprocesses it.
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            df_display = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['date_utc'])
            # Re-create df_model from df_display to ensure consistency with preprocessing steps
            # This is important because the model expects specific columns after one-hot encoding.
            # We need to re-apply the dummy encoding based on the same columns.
            features = ['payload_mass_kg', 'core_reused', 'year', 'rocket_name', 'launchpad_name']
            df_model = df_display[features + ['launch_success']].copy()
            df_model = pd.get_dummies(df_model, columns=['rocket_name', 'launchpad_name'], drop_first=True)
            df_model = df_model.fillna(0) # Fill with 0 for one-hot encoded columns
            for col in df_model.columns:
                if df_model[col].dtype == 'bool':
                    df_model[col] = df_model[col].astype(int)

            print(f"Loaded data from {PROCESSED_DATA_PATH}")
            return df_display, df_model
        except Exception as e:
            print(f"Error loading preprocessed data: {e}. Fetching new data.")
            pass # Fallback to fetching new data

    print("Fetching new SpaceX data...")
    raw_data = fetch_spacex_data()
    if raw_data:
        df_display, df_model = preprocess_data(raw_data)
        if not df_display.empty:
            # Create data directory if it doesn't exist
            os.makedirs(DATA_DIR, exist_ok=True)
            df_display.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"Saved preprocessed data to {PROCESSED_DATA_PATH}")
        return df_display, df_model
    else:
        print("Failed to fetch or preprocess data.")
        return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    # This block runs when data_fetcher.py is executed directly
    # It demonstrates fetching and preprocessing
    print("Running data_fetcher.py directly...")
    df_display, df_model = load_and_preprocess_data()
    if not df_display.empty:
        print("\nDisplay DataFrame Head:")
        print(df_display.head())
        print("\nModel DataFrame Head:")
        print(df_model.head())
        print("\nDataFrame Info:")
        df_display.info()
    else:
        print("No data loaded or processed.")
