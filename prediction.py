import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st

fastf1.Cache.enable_cache("f1_cache")

st.set_page_config(page_title="F1 Analysis & Prediction", layout="wide")

st.sidebar.title("Navigation")
tabs = st.sidebar.selectbox("Go to", ["Analysis & Visualization", "Race Prediction"])

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

    results = results.merge(races[['raceId', 'circuitId']], on='raceId', how='left')

    races['date'] = pd.to_datetime(races['date'])
    races.sort_values('date', ascending=False, inplace=True)
    races['days_since_last_race'] = races['date'].diff().dt.days

    st.sidebar.header('Analysis Sections')
    options = [
        'Top Drivers by Wins', 'Top Constructors by Wins', '1-2 Finishes', 'Podiums', 'Pole Positions', 
        'Circuits Analysis', 'Top Nationalities', 'Unluckiest Drivers',
        'Quali vs Race Performance', 'Pit Stop Duration of Last 20 Races', 'DNF Trends Across Circuits'
    ]
    selected_option = st.sidebar.selectbox('Select Analysis', options)


    if selected_option == 'Top Drivers by Wins':
        results['year'] = results['raceId'].map(races.set_index('raceId')['date'].dt.year)
        driver_wins = results.groupby('driverId')['positionOrder'].apply(lambda x: (x == 1).sum()).reset_index()
        driver_wins.columns = ['driverId', 'wins']
        top_drivers = driver_wins.sort_values(by='wins', ascending=False).head(10)
        top_drivers = top_drivers.merge(drivers[['driverId', 'surname']], on='driverId')

        st.subheader('Top 10 Drivers by Wins')
        st.dataframe(top_drivers)
        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='wins', data=top_drivers, palette='pastel', ax=ax)
        plt.title('Top 10 Drivers by Wins')
        st.pyplot(fig)


    if selected_option == 'Top Constructors by Wins':
        constructor_wins = results.groupby('constructorId')['positionOrder'].apply(lambda x: (x == 1).sum()).reset_index()
        constructor_wins.columns = ['constructorId', 'wins']
        top_teams = constructor_wins.sort_values(by='wins', ascending=False).head(10)
        top_teams = top_teams.merge(constructors[['constructorId', 'name']], on='constructorId')

        st.subheader('Top 10 Constructors by Wins')
        st.dataframe(top_teams)
        fig, ax = plt.subplots()
        sns.barplot(y='name', x='wins', data=top_teams, palette='viridis', ax=ax)
        plt.title('Top 10 Constructors by Wins')
        st.pyplot(fig)


    if selected_option == '1-2 Finishes':
        one_two_finishes = results.groupby(['raceId', 'constructorId'])['positionOrder'].apply(lambda x: set(x) == {1, 2}).reset_index()
        one_two_finishes = one_two_finishes[one_two_finishes['positionOrder'] == True]
        one_two_count = one_two_finishes['constructorId'].value_counts().reset_index()
        one_two_count.columns = ['constructorId', 'one_two_finishes']
        one_two_count = one_two_count.merge(constructors, on='constructorId')

        st.subheader('Top Constructors by 1-2 Finishes')
        st.dataframe(one_two_count)
        fig, ax = plt.subplots()
        sns.barplot(y='name', x='one_two_finishes', data=one_two_count.head(10), palette='inferno', ax=ax)
        plt.title('Top 10 Constructors with Most 1-2 Finishes')
        st.pyplot(fig)


    if selected_option == 'Podiums':
        podiums = results[results['positionOrder'].isin([1, 2, 3])]
        podium_count = podiums.groupby('driverId')['positionOrder'].count().reset_index()
        podium_count.columns = ['driverId', 'podiums']
        top_podium_drivers = podium_count.sort_values(by='podiums', ascending=False).head(10)
        top_podium_drivers = top_podium_drivers.merge(drivers[['driverId', 'surname']], on='driverId')

        st.subheader('Top 10 Drivers by Podiums')
        st.dataframe(top_podium_drivers)
        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='podiums', data=top_podium_drivers, palette='magma', ax=ax)
        plt.title('Top 10 Drivers by Podiums')
        st.pyplot(fig)


    if selected_option == 'Pole Positions':
        pole_positions = qualifying[qualifying['position'] == 1]
        pole_counts = pole_positions['driverId'].value_counts().reset_index()
        pole_counts.columns = ['driverId', 'poles']
        top_pole_drivers = pole_counts.merge(drivers[['driverId', 'surname']], on='driverId').head(10)

        st.subheader('Top 10 Drivers by Pole Positions')
        st.dataframe(top_pole_drivers)
        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='poles', data=top_pole_drivers, palette='cool', ax=ax)
        plt.title('Top 10 Drivers by Pole Positions')
        st.pyplot(fig)


    if selected_option == 'Circuits Analysis':
        circuit_wins = results.groupby('circuitId')['positionOrder'].apply(lambda x: (x == 1).sum()).reset_index()
        circuit_wins.columns = ['circuitId', 'wins']
        top_circuits = circuit_wins.sort_values(by='wins', ascending=False).head(10)
        top_circuits = top_circuits.merge(circuits[['circuitId', 'name']], on='circuitId')

        st.subheader('Top 10 Circuits by Wins')
        st.dataframe(top_circuits)
        fig, ax = plt.subplots()
        sns.barplot(y='name', x='wins', data=top_circuits, palette='plasma', ax=ax)
        plt.title('Top 10 Circuits by Wins')
        st.pyplot(fig)


    if selected_option == 'Top Nationalities':
        top_nationalities = drivers['nationality'].value_counts().reset_index()
        top_nationalities.columns = ['nationality', 'count']

        st.subheader('Top Nationalities of Drivers')
        st.dataframe(top_nationalities.head(10))
        fig, ax = plt.subplots()
        sns.barplot(y='nationality', x='count', data=top_nationalities.head(10), palette='crest', ax=ax)
        plt.title('Top Nationalities of Drivers')
        st.pyplot(fig)


    if selected_option == 'Unluckiest Drivers':
        results = results.merge(status[['statusId', 'status']], on='statusId', how='left')
        dnf_data = results[results['status'].str.contains('DNF|Accident|Collision|Engine|Gearbox|Retired|Mechanical', case=False, na=False)]
        unlucky_drivers = dnf_data['driverId'].value_counts().reset_index()
        unlucky_drivers.columns = ['driverId', 'dnf_count']
        unlucky_drivers = unlucky_drivers.merge(drivers[['driverId', 'surname']], on='driverId').head(10)

        st.subheader('Top 10 Unluckiest Drivers (Most DNFs)')
        st.dataframe(unlucky_drivers)
        fig, ax = plt.subplots()
        sns.barplot(y='surname', x='dnf_count', data=unlucky_drivers, palette='rocket', ax=ax)
        plt.title('Top 10 Unluckiest Drivers (Most DNFs)')
        st.pyplot(fig)


    if selected_option == 'Quali vs Race Performance':
        qualifying_summary = qualifying.groupby('driverId')['position'].mean().reset_index()
        qualifying_summary.columns = ['driverId', 'average_qualifying_position']

        race_summary = results.groupby('driverId')['positionOrder'].mean().reset_index()
        race_summary.columns = ['driverId', 'average_race_position']

        performance_comparison = qualifying_summary.merge(race_summary, on='driverId')
        performance_comparison = performance_comparison.merge(drivers[['driverId', 'surname']], on='driverId')

        st.subheader('Qualifying vs Race Performance')
        st.dataframe(performance_comparison.head(10))

        fig, ax = plt.subplots()
        sns.scatterplot(x='average_qualifying_position', y='average_race_position', data=performance_comparison)
        plt.title('Qualifying vs Race Performance')
        plt.xlabel('Average Qualifying Position')
        plt.ylabel('Average Race Position')
        st.pyplot(fig)


    if selected_option == 'Pit Stop Duration of Last 20 Races':
        recent_races = races.head(20)
        recent_pit_stops = pit_stops[pit_stops['raceId'].isin(recent_races['raceId'])]
        pit_stop_durations = recent_pit_stops.groupby('raceId')['milliseconds'].mean().reset_index()
        pit_stop_durations = pit_stop_durations.merge(races[['raceId', 'name']], on='raceId')

        st.subheader('Average Pit Stop Duration of Last 20 Races')
        st.dataframe(pit_stop_durations)

        fig, ax = plt.subplots()
        sns.lineplot(x='name', y='milliseconds', data=pit_stop_durations, marker='o', ax=ax)
        plt.xticks(rotation=90)
        plt.title('Pit Stop Duration of Last 20 Races')
        plt.ylabel('Average Pit Stop Duration (ms)')
        st.pyplot(fig)


elif tabs == "Race Prediction":
    st.title("🏁 F1 Race Prediction")

    session_2024 = fastf1.get_session(2024, "China", "R")
    session_2024.load()

    laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps_2024.dropna(inplace=True)

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

    sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

    qualifying_2025 = pd.DataFrame({
        "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
                   "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
                   "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
                   "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
        "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                               91.021, 91.079, 91.103, 91.638, 91.706,
                               91.625, 91.632, 91.688, 91.773, 91.840,
                               91.992, 92.018, 92.092, 92.141, 92.174]
    })

    driver_mapping = {
        "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
        "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
        "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
        "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
        "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
    }

    qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

    merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

    X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
    y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
    model.fit(X_train, y_train)

    predicted_race_times = model.predict(X)
    qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

    qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

    st.subheader("Predicted Race Results (2025 Chinese GP)")
    st.write(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")






