import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import arrow
import geopandas as gpd
from shapely.geometry import Point
from functools import lru_cache
from haversine import haversine_distance
from datetime import timedelta

def haversine_old(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def load_weather_data():
    return pd.read_csv('data/weather/city_info.csv')

def load_city_data(city_id):
    fpath = 'data/weather/' + str(city_id) + '.csv'
    return pd.read_csv(fpath)

def load_fire_data():
    return pd.read_csv('data/modis_2021_United_States.csv')

@lru_cache(maxsize=None)
def find_closest_city(lat, lon):
    weather_df = load_weather_data()
    weather_df['distance'] = haversine_vectorized(lat, lon, weather_df['Lat'], weather_df['Lon'])
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


def checkForFire_old(lat, lon, date, fire_df):
    date = pd.to_datetime(date)
    fire_df = fire_df[(fire_df['acq_date'] >= date - timedelta(days=1)) & (fire_df['acq_date'] <= date + timedelta(days=1))].copy()
    fire_df['distance'] = haversine_vectorized(lat, lon, fire_df['latitude'], fire_df['longitude'])
    closest_fire = fire_df.loc[fire_df['distance'].idxmin()]
    closest_dist = closest_fire['distance']
    closest_date = arrow.get(closest_fire['acq_date'])
    delta = abs(closest_date - arrow.get(date))
    if closest_dist < 10 and delta.days < 1:
        message = 'There has been a fire in this area. It occurred on ' + closest_fire['acq_date'] + ' and was ' + str(closest_dist) + ' km away from the specified coordinates. Coordinates: ' + str(closest_fire['latitude']) + ',' + str(closest_fire['longitude'])
        # print(message)
        return 1
    else:
        message = 'There has not been a fire in this area.'
        # print(message)
        return 0

def get_fire_path(date):
    year = date.strftime("%Y")
    path = "data/fire/modis_" + year + "_United_states.csv"
    return path

def checkForFire_oldd(lat, lon, date):
    date = pd.to_datetime(date)
    fire_df = pd.read_csv(get_fire_path(date))
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
    fire_df = fire_df[(fire_df['acq_date'] >= date - timedelta(days=1)) & (fire_df['acq_date'] <= date + timedelta(days=1))].copy()
    
    fire_gdf = gpd.GeoDataFrame(
        fire_df, geometry=gpd.points_from_xy(fire_df.longitude, fire_df.latitude), crs="EPSG:4326"
    )
    
    search_point = Point(lon, lat)
    
    # Create a spatial index
    spatial_index = fire_gdf.sindex
    
    # Find the nearest fire
    nearest_index = list(spatial_index.nearest((search_point,), 1))[0]
    nearest_fire = fire_gdf.iloc[nearest_index]
    nearest_lat = nearest_fire['latitude']
    nearest_lon = nearest_fire['longitude']
    nearest_dist = haversine_vectorized(np.array([lat]), np.array([lon]), np.array([nearest_lat]), np.array([nearest_lon]))[0]

    # Check if the fire is within 10 km
    if nearest_dist < 10:
        message = 'There has been a fire in this area. It occurred on ' + str(nearest_fire['acq_date']) + ' and was ' + str(nearest_dist) + ' km away from the specified coordinates. Coordinates: ' + str(nearest_lat) + ',' + str(nearest_lon)
        print(message)
        return 1
    else:
        message = 'There has not been a fire in this area.'
        return 0


def checkForFire(lat, lon, dates):
    fire_flags = np.zeros(len(dates), dtype=bool)
    for i, date in enumerate(dates):
        fire_flags[i] = 1 if checkForFire(lat, lon, date) else 0
    return fire_flags


# Load the data once
weather_data = load_weather_data()


# Example usage
lat = 40.833458
lon = -73.457279
date = '1869-01-01'
parseData(lat, lon, date)

