import pandas as pd

df = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
# Remove unuseful columns
df = df[['dateRep', 'cases', 'deaths', 'countriesAndTerritories', 'countryterritoryCode', 'continentExp']]
# Rename columns
df = df.rename(columns={
    'dateRep': 'date',
    'countriesAndTerritories': 'country',
    'countryterritoryCode': 'countryCode',
    'continentExp': 'continent'
})
# Convert string to datetime
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
# Preview the data frame
df.sample(10)
from datetime import datetime
# Get today as string
today = datetime.now().strftime('%Y-%m-%d')
# Get a data frame only for today
df_today = df[df.date == today]
# Preview the data frame
df_today.head()
import plotly.express as px
fig = px.scatter_geo(
    df_today, 
    locations='countryCode',
    color='continent',
    hover_name='country',
    size='cases',
    projection="natural earth",
    title=f'World COVID-19 Cases for {today}'
)
fig.show()
# Convert date to string type
df['date'] = df.date.dt.strftime('%Y%m%d')
# Sort the data frame on date
df = df.sort_values(by=['date'])
# Some countries does not have code, let's drop all the invalid rows
df = df.dropna()
# Preview the data frame
df.head(10)

fig = px.scatter_geo(
    df, 
    locations='countryCode',
    color='continent',
    hover_name='country',
    size='cases',
    projection="natural earth",
    title=f'World COVID-19 Cases',
    animation_frame="date"
)
fig.show()