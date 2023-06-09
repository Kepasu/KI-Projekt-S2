import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import arrow
from scipy.spatial.distance import cdist

fire_df = pd.read_csv('/users/emilychristians/desktop/ki_projekt/data/modis_2021_United_States.csv')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def load_weather_data():
    return pd.read_csv('/users/emilychristians/desktop/ki_projekt/data/weather/city_info.csv')

def load_city_data(city_id):
    fpath = '/users/emilychristians/desktop/ki_projekt/data/weather/' + str(city_id) + '.csv'
    return pd.read_csv(fpath)

def load_fire_data():
    return pd.read_csv('/users/emilychristians/desktop/ki_projekt/data/modis_2021_United_States.csv')

def find_closest_city(lat, lon):
    weather_df = load_weather_data()
    weather_df['distance'] = haversine(lat, lon, weather_df['Lat'], weather_df['Lon'])
    closest_city = weather_df.loc[weather_df['distance'].idxmin()]['ID']
    return closest_city

def find_closest_date(date, city_df):
    date_row = city_df.loc[city_df['Date'] == date]
    if not date_row.empty:
        return date_row.iloc[0]
    closest_date_idx = (pd.to_datetime(city_df['Date']) - pd.to_datetime(date)).abs().idxmin()
    return city_df.loc[closest_date_idx]

def parseData(lat, lon, date):
    closest_city_id = find_closest_city(lat, lon)
    city_df = load_city_data(closest_city_id)
    date_row = find_closest_date(date, city_df)
    if isinstance(date_row, pd.Series):
        tmax = date_row['tmax']
        tmin = date_row['tmin']
        prcp = date_row['prcp']
        return tmax, tmin, prcp
    else:
        return "Temperature data not found for the given date."

def parseTmax(lat, lon, date):
    closest_city_id = find_closest_city(lat, lon)
    city_df = load_city_data(closest_city_id)
    date_row = find_closest_date(date, city_df)
    if isinstance(date_row, pd.Series):
        temperature = date_row['tmax']
        if pd.isna(temperature):  # Check if temperature is NaN
            return 0
        else:
            return int(temperature)
    else:
        closest_temp = city_df.loc[date_row['date_idx']]['tmax']
        if pd.isna(closest_temp):  # Check if temperature is NaN
            return 0
        else:
            return int(closest_temp)


def parseTmin(lat, lon, date):
    closest_city_id = find_closest_city(lat, lon)
    city_df = load_city_data(closest_city_id)
    date_row = find_closest_date(date, city_df)
    if isinstance(date_row, pd.Series):
        temperature = date_row['tmin']
        if pd.isna(temperature):  # Check if temperature is NaN
            return 0
        else:
            return int(temperature)
    else:
        closest_temp = city_df.loc[date_row['date_idx']]['tmin']
        if pd.isna(closest_temp):  # Check if temperature is NaN
            return 0
        else:
            return int(closest_temp)


def parsePrcp(lat, lon, date):
    closest_city_id = find_closest_city(lat, lon)
    city_df = load_city_data(closest_city_id)
    date_row = find_closest_date(date, city_df)
    if isinstance(date_row, pd.Series):
        return date_row['prcp']
    else:
        return "Temperature data not found for the given date."


# Load the data once
weather_data = load_weather_data()
fire_data = load_fire_data()

lat = 30.006
lon = -155.2798
date = '2021-01-01'
#parseData(lat,lon,date)



def checkForFire(lat, lon, date, fire_df):
    fire_df['distance'] = haversine(lat, lon, fire_df['latitude'], fire_df['longitude'])
    closest_fire = fire_df.loc[fire_df['distance'].idxmin()]
    closest_dist = closest_fire['distance']
    closest_date = arrow.get(closest_fire['acq_date'])
    #delta = abs(closest_date - arrow.get(date))
    if closest_dist < 20:# and delta.days < 10:
        message = 'There has been a fire in this area. It occurred on ' + closest_fire['acq_date'] + ' and was ' + str(closest_dist) + ' km away from the specified coordinates. Coordinates: ' + str(closest_fire['latitude']) + ',' + str(closest_fire['longitude'])
        print(message)
        return True
    else:
        return False
