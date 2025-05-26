# SpaceX Launch Analysis & Prediction Platform

This project develops an advanced Python application that visualizes SpaceX launch data and predicts the success of future launches using machine learning models. The platform offers interactive dashboards, detailed analytics, and predictive insights based on historical data.

## Project Structure


spacex_app/
├── app.py
├── data_fetcher.py
├── model_trainer.py
├── requirements.txt
└── README.md


## Setup and Installation

1.  **Clone the repository (or create the files manually):**
    If you have this as a repository, clone it. Otherwise, create the `spacex_app` directory and place the provided files (`app.py`, `data_fetcher.py`, `model_trainer.py`, `requirements.txt`, `README.md`) inside it.

2.  **Navigate to the project directory:**
    ```bash
    cd spacex_app
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

4.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Ensure you are in the `spacex_app` directory and your virtual environment is activated.**

2.  **Run the `model_trainer.py` script first to fetch data and train the model:**
    This step will download the necessary data from the SpaceX API, preprocess it, train the machine learning model, and save the trained model and feature columns to the `models/` directory. It will also save the processed data to the `data/` directory.
    ```bash
    python model_trainer.py
    ```
    You should see messages indicating data fetching, preprocessing, and model training/saving.

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This command will open the application in your web browser.

## Features

* **Historical Launch Data:** View a table of past SpaceX launches with filters for year and launch site.
* **Exploratory Data Analysis (EDA):** Interactive charts showing launch success rates by rocket type and launch site, and payload mass distribution.
* **Geospatial Map:** A map visualizing launch sites and the outcomes of launches.
* **Predictive Tool:** Input parameters like payload mass, rocket type, launch site, and core reusability to get a predicted probability of launch success.

## Data Sources

* **SpaceX-API:** Used to fetch historical launch data.
    * [SpaceX-API Documentation](https://github.com/r-spacex/SpaceX-API/blob/master/docs/v4/README.md)

## Machine Learning Model

A classification model (Logistic Regression and Random Forest) is used to predict launch success based on historical data. The model is trained on features such as payload mass, rocket type, launch site, and core reusability.

## Developed By

Syed Ahmed Ali | Powered by Machine Learning
