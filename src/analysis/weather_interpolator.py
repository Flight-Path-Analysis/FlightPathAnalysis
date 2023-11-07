import numpy as np
import pandas as pd
import os
import math

from src.common import utils
from src.models import weather

class WeatherInterpolator:
    def __init__(self, config):
        self.config = config

    def gaussian_interpolation(self, target, data, quantity):
        """
        Perform a Gaussian-weighted interpolation for a specified quantity based on geographic proximity.

        This function computes distances between a target location and a set of locations in the data,
        then applies a Gaussian weighting based on these distances and the 'sigma' values of the stations.
        The result is a weighted average of the specified quantity, considering the influence of each
        station's data based on its spatial relationship to the target point.

        Parameters:
        target (dict): A dictionary representing the target point, containing 'lat' and 'lon' keys with
                    geographical coordinates.
        data (pd.DataFrame): A DataFrame containing station data, each row representing a station. It must
                            include 'lat', 'lon', and 'sigma' columns, representing the geographical
                            coordinates of the station and the standard deviation of the Gaussian distribution
                            used for weighting, respectively. The DataFrame also contains a column corresponding
                            to the 'quantity' parameter that holds the values to be interpolated.
        quantity (str): The name of the column in 'data' that represents the quantity to be interpolated. This
                        column's values are numerically interpolated.

        Returns:
        float: The Gaussian-weighted interpolated value of the specified quantity at the target location.
        """

        distances = utils.haversine_distance(target['lat'], target['lon'], data['lat'], data['lon'])
        # If the target is too far way from any station (5 sigma), the measure is not reliable.
        if min(distances) > 3*np.mean(data['sigma']):
            return np.nan
        weights = np.exp(-distances**2/(2*data['sigma']**2/10))
        weights = weights/np.sum(weights)
        avg = np.sum(weights*data[quantity])
        return avg

    def load_relevant_files(self, time):
        # Calculate the start and end times of interest
        time_thresh = self.config['statistics']['interpolation']['weather']['time-thresh']
        start_time = time - time_thresh
        end_time = time + time_thresh

        # Directory where the weather files are stored
        weather_dir = self.config['data-gather']['weather']['out-dir']

        # Initialize an empty DataFrame to hold all the relevant data
        relevant_data = pd.DataFrame()

        # Iterate over all files in the directory
        for file in os.listdir(weather_dir):
            # Check if file matches the expected 'start_end.csv' pattern
            if '_' in file and file.endswith('.csv'):
                # Extract start and end times from the filename
                file_start, file_end = map(int, file.rstrip('.csv').split('_'))
                # Check if the file's time range overlaps with the desired time range
                if not (file_end < start_time or file_start > end_time):
                    # If it overlaps, load the file and concatenate it to the DataFrame
                    file_path = os.path.join(weather_dir, file)
                    file_data = pd.read_csv(file_path)
                    relevant_data = pd.concat([relevant_data, file_data])

        # Return the concatenated DataFrame containing all the relevant data
        return relevant_data

    def estimate_scalar(self, target, scalar):

        relevant_data = self.load_relevant_files(target['time'])
        relevant_data = weather.calibrate_stations(relevant_data)
        relevant_data['f{scalar}_elev'] = [row[f'{scalar}_model'].predict([target['elevation']])[0] for _, row in relevant_data.iterrows()]
        relevant_data = relevant_data[['station', 'lat', 'lon', 'elevation', 'timestamp', f'{scalar}_elev', 'sigma']].copy()
        relevant_data = relevant_data[abs(relevant_data['timestamp'] - target['timestamp']) <= 3600/2]
        relevant_data = relevant_data.dropna(subset=[f'{scalar}_elev'])
        relevant_data = relevant_data.groupby('station').agg({
            'lon': 'mean',
            'lat': 'mean',
            'elevation': 'mean',
            'timestamp': 'mean',
            f'{scalar}_elev': 'mean',
            'sigma': 'mean'
        })
        scalar = self.gaussian_interpolation(target, relevant_data, f'{scalar}_elev')

        return scalar

    def estimate_flight_wind(self, flight):
        dx = [utils.haversine_distance(row_a['lat'], row_a['lon'], row_b['lat'], row_b['lon']) for row_a, row_b in zip(flight[:-1].itertuples(), flight[1:].itertuples())]
        dt = [row_b['time'] - row_a['time'] for row_a, row_b in zip(flight[:-1].itertuples(), flight[1:].itertuples())]
        v = dx/dt*1.94384 #m/s to knots

        bearings = [utils.haversine_bearing(row_a['lat'], row_a['lon'], row_b['lat'], row_b['lon']) for row_a, row_b in zip(flight[:-1].itertuples(), flight[1:].itertuples())]

        flight['geo_velocity'] = v + [v[-1]]
        flight['geo_heading'] = bearings + [bearings[-1]]

        geo_heading_rad = math.radians(flight['geo_heading'])
        heading_rad = math.radians(flight['geo_heading'])

        # Compoent of the geo velocity in the direction of the heading.
        geo_v_h = flight['geo_velocity']*np.cos(heading_rad - geo_heading_rad)
        # Component of the wind vector in the direction of the heading.
        wind_v_g = geo_v_h - flight['velocity']
        
        flight['wind_velocity_heading'] = wind_v_g
        
        return flight

    def compute_flight_weather_quantities(self, flight):
        temperatures = np.zeros(len(flight))
        pressures = np.zeros(len(flight))
        densities = np.zeros(len(flight))
        clouds = np.zeroes(len(flight))

        for i, row in flight.iterrows():
            target = {
                'time': row['timestamp'],
                'lat': row['latitude'],
                'lon': row['longitude'],
                'elevation': row['altitude']
            }
            temperatures[i] = self.estimate_scalar(target, 'tmpf')
            pressures[i] = self.estimate_scalar(target, 'air_pressure')
            densities[i] = self.estimate_scalar(target, 'air_density')
            clouds[i] = self.estimate_scalar(target, 'clouds')
            
        flight['tmpf'] = temperatures
        flight['air_pressure'] = pressures
        flight['air_density'] = densities
        flight['clouds'] = clouds
            
        flight = self.estimate_flight_wind(flight)

        return flight
