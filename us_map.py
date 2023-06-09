import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import earthpy as et
import geodatasets
from searchdata import checkForFire

#weather_df = pd.read_csv('/Users/emilychristians/desktop/data/weather/city_info.csv')
fire_df = pd.read_csv('/Users/emilychristians/desktop/ki_projekt/data/modis_2021_United_States.csv')

# Adjust plot font sizes
sns.set(font_scale=1.5)
sns.set_style("white")

# Set working dir & get data
data = et.data.get_data('spatial-vector-lidar')
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

# Import world boundary shapefile
worldBound_path = os.path.join("data", "spatial-vector-lidar", "global", 
                               "ne_110m_land", "ne_110m_land.shp", "/Users/emilychristians/desktop/ki_projekt/data/cb_2014_us_state_500k/cb_2014_us_state_500k.shp")
worldBound = gpd.read_file(worldBound_path)

# Create numpy array of x,y point locations
#longitude then latitude 
add_points = np.array([[-105.2519,   40.0274], 
                       [ -80.2102,   25.7839]])

# Turn points into list of x,y shapely points 
city_locations = [Point(xy) for xy in add_points]
city_locations

# Create geodataframe using the points
city_locations = gpd.GeoDataFrame(city_locations, 
                                  columns=['geometry'],
                                  crs=worldBound.crs)
city_locations.head(3)

# Plot point locations
fig, ax = plt.subplots(figsize=(10, 7))

worldBound.plot(figsize=(10, 5), color='k',
               ax=ax)
# Add city locations
city_locations.plot(ax=ax, 
                    color='red', # was originally springgreen
                    marker='.',
                    markersize=70)

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

def onclick(event):
    print('%s click: button=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
            event.xdata, event.ydata))
    checkForFire(event.ydata, event.xdata, '2023-05-25', fire_df)
    print(checkForFire)

#display results on the map/ next to it. noes not work yet 
#plt.text(-120, 56, return onclick, fontsize = 22, 
#    bbox = dict(facecolor = 'moccasin', alpha = 0.5))
#plt.plot(c = 'g')


cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Add title and axes labels
ax.set(title="World Map - Geographic Coordinate Reference System (long/lat degrees)",
       xlabel="X Coordinates (latitude)",
       ylabel="Y Coordinates (longitude)");

plt.show()
