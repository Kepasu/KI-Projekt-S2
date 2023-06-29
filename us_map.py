import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import earthpy as et
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import arrow
plt.style.use('seaborn-white')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def combine_fire_df(start_year, end_year):
    years = range(start_year, end_year + 1)
    file_paths = [os.path.join('/Users/emilychristians/desktop/ki_projekt/data/modis_data/',
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
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date']).dt.normalize()
    weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.normalize()

    fire_df = fire_df.loc[:, ['latitude', 'longitude', 'acq_date']]

    merged_df = pd.merge(weather_df, fire_df, left_on='Date',
                         right_on='acq_date', how='left')
    merged_df['Fire'] = merged_df['acq_date'].notnull().astype(int)
    merged_df = merged_df.drop(['latitude', 'longitude', 'acq_date'], axis=1)

    return merged_df


def checkForFire(lat, lon, date, fire_df):
    fire_df['distance'] = haversine(lat, lon, fire_df['latitude'], fire_df['longitude'])
    closest_fire = fire_df.loc[fire_df['distance'].idxmin()]
    closest_dist = closest_fire['distance']
    closest_date = arrow.get(closest_fire['acq_date'])
    if closest_dist < 20:
        message = 'There has been a fire in this area.\nIt occurred on ' + closest_fire[
            'acq_date'] + ' and was \n' + str(closest_dist) + '\nkm away from the specified coordinates. \nCoordinates: ' + str(
            closest_fire['latitude']) + '\n,' + str(closest_fire['longitude'])
        if ax.texts:
            ax.texts[0].remove()  # Remove previous annotation if exists
        plt.annotate(message, xy=(closest_fire['longitude'], closest_fire['latitude']), xytext=(0, -10),
                     textcoords='offset points', arrowprops=dict(facecolor='forestgreen', arrowstyle='->'),
                     color='forestgreen', backgroundcolor='white', fontsize=10)  # Set fontsize to 10
        fig.canvas.draw()
        return True
    else:
        return False


# Load weather and fire dataframes
weather_df = pd.read_csv('/Users/emilychristians/desktop/ki_projekt/data/weather/city_info.csv')
fire_df = combine_fire_df(2001, 2022)

# Set the path to the 'city_info.csv' file
city_info_path = '/Users/emilychristians/desktop/ki_projekt/data/weather/city_info.csv'


# Read the 'city_info.csv' file into a pandas DataFrame
city_info_df = pd.read_csv(city_info_path)

# Adjust plot font sizes
sns.set(font_scale=1.5)
sns.set_style("white")

# Set working dir & get data
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

# Import world boundary shapefile
worldBound_path = os.path.join("data", "spatial-vector-lidar", "global", "ne_110m_land", "ne_110m_land.shp", "/Users/emilychristians/Desktop/ki_projekt/data/cb_2014_us_state_500k/cb_2014_us_state_500k.shp")
worldBound = gpd.read_file(worldBound_path)

# Create a GeoDataFrame from the city_info_df DataFrame
geometry = [Point(xy) for xy in zip(city_info_df['Lon'], city_info_df['Lat'])]
city_locations = gpd.GeoDataFrame(city_info_df, geometry=geometry)

# Plot point locations
fig, ax = plt.subplots(figsize=(10, 7))

worldBound.plot(figsize=(10, 5), color='k', ax=ax)

# Add city locations
city_locations.plot(ax=ax, color='mediumturquoise', marker='.', markersize=50)

# Setup x y axes with labels and add graticules
ax.set(xlabel="Longitude (Degrees)", ylabel="Latitude (Degrees)",
       title="World Map - PROBABILITY OF FIRE - WGS84 Datum\n Units: Degrees - Latitude / Longitude")
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')

# Zooming into Main Land USA
x1, x2, y1, y2 = -126, -65, 23, 50
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)

ax.indicate_inset_zoom(ax, edgecolor="black")

# Contour Plot --> make the fire data appear as a colored third dimension
x = np.arange(-126, -60, 1)
y = np.arange(23, 55, 0.9)
X, Y = np.meshgrid(x, y)

# Calculate fire probability (example data)
fire_prob = np.random.rand(len(y), len(x))

# Plot contour lines
contour = ax.contour(X, Y, fire_prob, cmap='hot', levels=10)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
cbar.set_label('Fire Probability')

def onclick(event):
    print('%s click: button=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.xdata, event.ydata))
    checkForFire(event.ydata, event.xdata, '2023-05-25', fire_df)
    print(checkForFire)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Add title and axes labels
ax.set(title="US Map - Probability of Fire (0-1)",
       xlabel="X Coordinates (latitude)",
       ylabel="Y Coordinates (longitude)")

plt.show()
