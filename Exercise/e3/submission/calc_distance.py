import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
from xml.dom.minidom import parse, parseString
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


#define get_data function
#https://docs.python.org/3/library/xml.dom.minidom.html
def get_data(file):
    gpx_file = parse(file)
    lat_long = gpx_file.getElementsByTagName('trkpt')
    lats = []
    longs = []
    
    for lat in lat_long:
        lats.append(lat.attributes['lat'].value)
        
    for long in lat_long:
        longs.append(long.attributes['lon'].value)
    
    temp_dict = {'lat': lats, 'lon': longs}
    lat_long_df= pd.DataFrame(temp_dict, columns = ['lat', 'lon'])
    
    lat_long_df['lat'] = lat_long_df['lat'].astype(float)
    lat_long_df['lon'] = lat_long_df['lon'].astype(float)
    return lat_long_df

#adapted from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas?noredirect=1&lq=1
def haversine_np(data):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = data.lat, data.lat2, data.lon, data.lon2

    lon1 = pd.to_numeric(lon1, errors='coerce')
    lon2 = pd.to_numeric(lon2, errors='coerce')
    lat1 = pd.to_numeric(lat1, errors='coerce')
    lat2 = pd.to_numeric(lat2, errors='coerce')
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km/1000

def distance(file):
    df = file
    df['lat2'] = df['lat'].shift(-1)
    df['lon2'] = df['lon'].shift(-1)
    df = df[:-1]
    #adapted from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas?noredirect=1&lq=1
    result = haversine_np(df)
    return result.sum()

def testhaversine():
     points = pd.DataFrame({
    'lat': [49.28, 49.26, 49.26],
    'lon': [123.00, 123.10, 123.05]})
     print(distance(points))

def smooth(data):
    kalman_data = data[['lat', 'lon']]
    
    
    initial_state = kalman_data.iloc[0]
    #20 meters can be error
    observation_covariance = np.diag([0.15, 0.15])**2
    transition_covariance = np.diag([0.001, 0.001])**2
    transition_matrix = np.diag([1,1])
    
    transition_covariance = np.diag([0.1, 0.1])**2
    kf = KalmanFilter(initial_state_mean = initial_state,
                        initial_state_covariance= observation_covariance,
                         observation_covariance = observation_covariance,
                         transition_covariance = transition_covariance,
                         transition_matrices = transition_matrix)
    
    kalman_smoothed, _ = kf.smooth(kalman_data)
    
    kalman_df = pd.DataFrame(data = kalman_smoothed[:], columns=['lat', 'lon'])


    df = kalman_df[:-1]
    return df
    
    
    
def main():
    #Create a data frame
    points = get_data(sys.argv[1])
    #points = get_data("walk1.gpx")
    #print(points)
    #print(points.dtype)
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
