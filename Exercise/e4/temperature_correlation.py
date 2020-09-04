import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from difflib import get_close_matches
import math


#adapter from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
#but changed
def haversine(lat1, lon1, lat2, lon2):
    R = 6371    # radius of the earth in km

    dLat = np.deg2rad(lat2-lat1)
    dLon = np.deg2rad(lon2-lon1)

    a = math.sin(dLat/2)* math.sin(dLat/2) + math.cos(math.radians(lat1))* math.cos(math.radians(lat2))* math.sin(dLon/2)* math.sin(dLon/2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R*c
    return d
haversine_func = np.vectorize(haversine)

def distance(city_data, stations):
    stations['distance'] = haversine_func(city_data['latitude'], city_data['longitude'], stations['latitude'], stations['longitude'])
    return stations['distance']


def best_tmax(city_data, stations):
    stations['distance'] = distance(city_data, stations)
    return stations['avg_tmax'][stations['distance'].idxmin()]



file1 = sys.argv[1] #stations.json.gz
#file1 = "stations.json.gz"
file2 = sys.argv[2] #city_data.csv
#file2 = "city_data.csv"
#file3 = "output.svg"
file3 = sys.argv[3] #output.svg

stations = pd.read_json(file1, lines = True)
city_data = pd.read_csv(file2, sep=",")
stations['avg_tmax'] = stations['avg_tmax'].div(10)
#print(stations)
# adapted from https://kite.com/python/answers/how-to-drop-empty-rows-from-a-pandas-dataframe-in-python#:~:text=Use%20df.,contain%20NaN%20under%20those%20columns.
city_data.dropna(subset = ["area"], inplace=True)
city_data.dropna(subset = ["population"], inplace=True)
#convert m^2 to km^2
city_data['area'] = city_data['area'].div(1000000)
dcity_data = city_data[(city_data['area']<10000)]
#print(city_data)

#Entity Resolution
city_data['density'] = city_data.population/city_data.area
city_data['best_tmax'] = 0

city_data['best_tmax'] = city_data.apply(best_tmax, 1, stations=stations)

plt.plot(city_data['best_tmax'], city_data['density'], 'b.')
plt.title('Relationship between Temperature and Population Density')
plt.xlabel('Avg Max Temperature (\u00b0C)')
plt.ylabel('Population Density (people/km\u00b2)')
plt.savefig(file3)
