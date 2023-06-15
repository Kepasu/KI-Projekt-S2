import pandas as pd
import numpy as np
import os
import pickle


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(
            np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def combine_fire_df(start_year, end_year):
    years = range(start_year, end_year + 1)
    file_paths = [os.path.join('data/fire/',
                               f"modis_{year}_United_States.csv")
                  for year in years]

    fire_df = pd.DataFrame()
    fire_df = pd.concat((pd.read_csv(file) for file in file_paths),
                        ignore_index=True)
    return fire_df


def filter_fire_df(lat, lon, fire_df):
    fire_df['distance'] = haversine(lat, lon,
                                    fire_df['latitude'], fire_df['longitude'])
    close_fire_df = fire_df[fire_df['distance'] <= 10]
    return close_fire_df


def check_for_fire(lat, lon, fire_df, weather_df):
    # only keep rows close to given date, keep only relevant columns
    # and convert 'acq_date' to datetime format
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date']).dt.normalize()
    weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.normalize()

    fire_df = fire_df.loc[:, ['latitude', 'longitude', 'acq_date']]

    merged_df = pd.merge(weather_df, fire_df, left_on='Date',
                         right_on='acq_date', how='left')
    merged_df['Fire'] = merged_df['acq_date'].notnull().astype(int)
    merged_df = merged_df.drop(['latitude', 'longitude', 'acq_date'], axis=1)

    return merged_df


def read_city_info():
    df = pd.read_csv('data/weather/city_info.csv')
    return df


def read_city_df(id, start_year, end_year):
    df = pd.read_csv('data/weather/' + id + '.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    start_date = pd.to_datetime(str(start_year))
    end_date = pd.to_datetime(str(end_year))

    filtered_df = df[(df['Date'] >= start_date)
                     & (df['Date'] <= end_date)]
    return filtered_df


def city_dataframes(start_year, end_year):
    info_df = read_city_info()
    fire_df = combine_fire_df(start_year, end_year)

    city_dfs = {}

    for index, row in info_df.iterrows():

        city_id = row['ID']
        city_df = read_city_df(city_id, start_year, end_year)

        # filter_df_years(city_df, start_year, end_year)
        new_fire_df = filter_fire_df(row['Lat'], row['Lon'], fire_df)

        df = check_for_fire(row['Lat'], row['Lon'],
                            new_fire_df, city_df)

        city_dfs.setdefault(city_id, [])
        city_dfs[city_id].append(df)
        city_dfs[city_id].append(row['Lat'])
        city_dfs[city_id].append(row['Lon'])

    return city_dfs


city_dfs = city_dataframes(2018, 2020)
# np.save("city_dict", city_dfs)
print(city_dfs.values())
with open('city_dict.pkl', 'wb') as f:
    pickle.dump(city_dfs, f)
