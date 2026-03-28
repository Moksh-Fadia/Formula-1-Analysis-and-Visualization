import os
import fastf1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Enable cache
cache_dir = os.path.expanduser("~/fastf1_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Streamlit page config
st.set_page_config(page_title="F1 Analysis & Prediction", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.selectbox("Go to", ["Analysis & Visualization", "Race Prediction"])

# Functions for prediction section
@st.cache_data
def load_laps():
    session = fastf1.get_session(2024, 8, "R")
    session.load()

    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(inplace=True)

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()

    return laps

@st.cache_resource
def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=39,
        monotone_constraints='(-1,-1,-1)'
    )
    model.fit(X_train, y_train)
    return model

# ==================== ANALYSIS SECTION ====================
if tabs == "Analysis & Visualization":
    st.title("📊 F1 Analysis & Visualization")

    races = pd.read_csv('races.csv')
    drivers = pd.read_csv('drivers.csv')
    constructors = pd.read_csv('constructors.csv')
    results = pd.read_csv('results.csv')
    qualifying = pd.read_csv('qualifying.csv')
    circuits = pd.read_csv('circuits.csv')
    driver_standings = pd.read_csv('driver_standings.csv')
    status = pd.read_csv('status.csv')
    pit_stops = pd.read_csv('pit_stops.csv')

    # preprocessing
    races['date'] = pd.to_datetime(races['date'])
    races.sort_values('date', ascending=False, inplace=True)

    results = results.merge(races[['raceId', 'circuitId']], on='raceId', how='left')

    qualifying['position'] = pd.to_numeric(qualifying['position'], errors='coerce')
    pit_stops['milliseconds'] = pd.to_numeric(pit_stops['milliseconds'], errors='coerce')

    options = [
        'Top Drivers by Wins',
        'Top Constructors by Wins',
        '1-2 Finishes',
        'Podiums',
        'Pole Positions',
        'Unluckiest Drivers',
        'Quali vs Race Performance',
        'Pit Stop Duration of Last 20 Races'
    ]

    selected_option = st.sidebar.selectbox('Select Analysis', options)

    # Top Drivers
    if selected_option == 'Top Drivers by Wins':
        driver_wins = results.groupby('driverId')['positionOrder'].apply(lambda x: (x == 1).sum()).reset_index()
        driver_wins.columns = ['driverId', 'wins']

        top_drivers = driver_wins.merge(drivers[['driverId', 'surname']], on='driverId') \
                                 .sort_values(by='wins', ascending=False).head(10)

        st.subheader('Top 10 Drivers by Wins')
        st.write(top_drivers)

        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='wins', data=top_drivers, palette='pastel', ax=ax)
        st.pyplot(fig)

    # Top Constructors
    elif selected_option == 'Top Constructors by Wins':
        constructor_wins = results.groupby('constructorId')['positionOrder'].apply(lambda x: (x == 1).sum()).reset_index()
        constructor_wins.columns = ['constructorId', 'wins']

        top_teams = constructor_wins.merge(constructors[['constructorId', 'name']], on='constructorId') \
                                   .sort_values(by='wins', ascending=False).head(10)

        st.subheader('Top 10 Constructors by Wins')
        st.write(top_teams)

        fig, ax = plt.subplots()
        sns.barplot(y='name', x='wins', data=top_teams, palette='viridis', ax=ax)
        st.pyplot(fig)

    # 1-2 Finishes
    elif selected_option == '1-2 Finishes':
        one_two = results[results['positionOrder'].isin([1, 2])]

        one_two_counts = one_two.groupby(['raceId', 'constructorId'])['positionOrder'].nunique().reset_index()
        one_two_counts = one_two_counts[one_two_counts['positionOrder'] == 2]

        one_two_summary = one_two_counts.groupby('constructorId').size().reset_index(name='count')

        one_two_summary = one_two_summary.merge(
            constructors[['constructorId', 'name']],
            on='constructorId'
        ).sort_values(by='count', ascending=False).head(10)

        st.subheader('Top 1-2 Finishes')
        st.write(one_two_summary)

        fig, ax = plt.subplots()
        sns.barplot(y='name', x='count', data=one_two_summary, palette='magma', ax=ax)
        st.pyplot(fig)

    # Podiums
    elif selected_option == 'Podiums':
        podiums = results[results['positionOrder'] <= 3]

        podium_counts = podiums.groupby('driverId').size().reset_index(name='podiums')

        podium_counts = podium_counts.merge(
            drivers[['driverId', 'surname']],
            on='driverId'
        ).sort_values(by='podiums', ascending=False).head(10)

        st.subheader('Top 10 Podiums')
        st.write(podium_counts)

        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='podiums', data=podium_counts, palette='coolwarm', ax=ax)
        st.pyplot(fig)

    # Pole Positions
    elif selected_option == 'Pole Positions':
        poles = qualifying[qualifying['position'] == 1]

        pole_counts = poles.groupby('driverId').size().reset_index(name='poles')

        pole_counts = pole_counts.merge(
            drivers[['driverId', 'surname']],
            on='driverId'
        ).sort_values(by='poles', ascending=False).head(10)

        st.subheader('Top Pole Positions')
        st.write(pole_counts)

        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='poles', data=pole_counts, ax=ax)
        st.pyplot(fig)

    # Unluckiest Drivers
    elif selected_option == 'Unluckiest Drivers':
        results_status = results.merge(status, on='statusId', how='left')

        dnf = results_status[~results_status['status'].str.contains('Finished', na=False)]

        dnf_counts = dnf.groupby('driverId').size().reset_index(name='dnfs')
        races_count = results.groupby('driverId').size().reset_index(name='races')

        unlucky = dnf_counts.merge(races_count, on='driverId')
        unlucky['dnf_ratio'] = unlucky['dnfs'] / unlucky['races']

        unlucky = unlucky.merge(
            drivers[['driverId', 'surname']],
            on='driverId'
        ).sort_values(by='dnf_ratio', ascending=False).head(10)

        st.subheader('Unluckiest Drivers (DNF Ratio)')
        st.write(unlucky[['surname', 'dnfs', 'races', 'dnf_ratio']])

        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='dnf_ratio', data=unlucky, palette='rocket', ax=ax)
        st.pyplot(fig)

    # Quali vs Race
    elif selected_option == 'Quali vs Race Performance':
        quali = qualifying[['raceId', 'driverId', 'position']].copy()
        quali.rename(columns={'position': 'quali_position'}, inplace=True)

        # Convert to numeric 
        quali['quali_position'] = pd.to_numeric(quali['quali_position'], errors='coerce')

        merged = results.merge(quali, on=['raceId', 'driverId'], how='inner')

        merged = merged.dropna(subset=['quali_position', 'positionOrder'])
        merged['gain'] = merged['quali_position'] - merged['positionOrder']

        performance = merged.groupby('driverId')['gain'].mean().reset_index()

        performance = performance.merge(
            drivers[['driverId', 'surname']],
            on='driverId'
        ).sort_values(by='gain', ascending=False).head(10)

        st.subheader('Best Race Performers (Avg Position Gain)')
        st.write(performance)

        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='gain', data=performance, palette='cubehelix', ax=ax)
        st.pyplot(fig)

    # Pit Stops
    elif selected_option == 'Pit Stop Duration of Last 20 Races':
        recent_races = races.head(20)['raceId']

        recent_pits = pit_stops[pit_stops['raceId'].isin(recent_races)]

        pit_avg = recent_pits.groupby('driverId')['milliseconds'].mean().reset_index()

        pit_avg = pit_avg.merge(
            drivers[['driverId', 'surname']],
            on='driverId'
        ).sort_values(by='milliseconds').head(10)

        st.subheader('Fastest Pit Stops (Last 20 Races)')
        st.write(pit_avg)

        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='milliseconds', data=pit_avg, palette='flare', ax=ax)
        st.pyplot(fig)


# ==================== RACE PREDICTION SECTION ====================
elif tabs == "Race Prediction":
    st.title("🏁 Monaco GP 2025 - Race Time Prediction")

    with st.spinner("Running prediction..."):

        # loading data
        laps_2024 = load_laps()

        # sector times
        sector_times_2024 = laps_2024.groupby("Driver").agg({
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean",
            "Sector3Time (s)": "mean"
        }).reset_index()

        sector_times_2024["TotalSectorTime (s)"] = sector_times_2024[
            ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
        ].sum(axis=1)

        clean_air_race_pace = {
            "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
            "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
            "OCO": 95.682128
        }

        team_points = {
            "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
            "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
        }

        max_points = max(team_points.values())
        team_perf = {team: pts / max_points for team, pts in team_points.items()}

        driver_to_team = {
            "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
            "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
            "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
        }

        qualifying_2025 = pd.DataFrame({
            "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
                       "HAM", "STR", "GAS", "ALO", "HUL"],
            "QualifyingTime (s)": [70.669, 69.954, 70.129, None, 71.362, 71.213, 70.063,
                                  70.942, 70.382, 72.563, 71.994, 70.924, 71.596]
        })

        qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)
        qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
        qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_perf)

        merged_data = qualifying_2025.merge(
            sector_times_2024[["Driver", "TotalSectorTime (s)"]],
            on="Driver", how="left"
        )

        # Keep only drivers present in laps
        valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
        merged_data = merged_data[valid_drivers]

        X = merged_data[[
            "QualifyingTime (s)",
            "TeamPerformanceScore",
            "CleanAirRacePace (s)"
        ]]

        y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
        y = y.fillna(y.median())  

        # preprocessing
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.3, random_state=37
        )

        model = train_model(X_train, y_train)

        merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

        final_results = merged_data.sort_values("PredictedRaceTime (s)")

        st.subheader("Predicted Monaco GP 2025 Results")
        st.dataframe(final_results[["Driver", "PredictedRaceTime (s)"]])

        y_pred = model.predict(X_test)
        st.write(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")





