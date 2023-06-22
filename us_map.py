import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import earthpy as et
import arrow

fire_df = pd.read_csv('/Users/emilychristians/desktop/ki_projekt/data/modis_2021_United_States.csv')

# Set the path to the 'city_info.csv' file
city_info_path = '/Users/emilychristians/desktop/ki_projekt/data/weather/city_info.csv'

# Read the 'city_info.csv' file into a pandas DataFrame
city_info_df = pd.read_csv(city_info_path)

# Adjust plot font sizes
sns.set(font_scale=1.5)
sns.set_style("white")

# Set working dir & get data
data = et.data.get_data('spatial-vector-lidar')
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

# Import world boundary shapefile
worldBound_path = os.path.join("data", "spatial-vector-lidar", "global",
                               "ne_110m_land", "ne_110m_land.shp",
                               "/Users/emilychristians/desktop/ki_projekt/data/cb_2014_us_state_500k/cb_2014_us_state_500k.shp")
worldBound = gpd.read_file(worldBound_path)

# Create a GeoDataFrame from the city_info_df DataFrame
geometry = [Point(xy) for xy in zip(city_info_df['Lon'], city_info_df['Lat'])]
city_locations = gpd.GeoDataFrame(city_info_df, geometry=geometry)

# Plot point locations
fig, ax = plt.subplots(figsize=(10, 7))
worldBound.plot(figsize=(10, 5), color='k', ax=ax)
city_locations.plot(ax=ax, color='red', marker='o', markersize=50)

# Setup x y axes with labels and add graticules
ax.set(xlabel="Longitude (Degrees)", ylabel="Latitude (Degrees)",
       title="Global Map - Geographic Coordinate System - WGS84 Datum\n Units: Degrees - Latitude / Longitude")
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')

# Zooming into Main Land USA
x1, x2, y1, y2 = -126, -65, 23, 50
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)

ax.indicate_inset_zoom(ax, edgecolor="black")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def load_city_data(city_id):
    fpath = '/users/emilychristians/desktop/ki_projekt/data/weather/' + str(city_id) + '.csv'
    return pd.read_csv(fpath)

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

def onclick(event):
    # if ctrl pressed, show city data
    # if event.ctrl ... get nearest city ... 
    # if not, show fire data
    print('%s click: button=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.xdata, event.ydata))
    checkForFire(event.ydata, event.xdata, '2023-05-25', fire_df)

# Connect the onclick function to the figure canvas
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
