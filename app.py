import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(layout="wide")
st.title('Formula 1 Data Analysis')

# Data Import
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

# Data Preprocessing
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
