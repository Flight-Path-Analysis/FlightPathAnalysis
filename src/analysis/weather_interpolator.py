import numpy as np
import pandas as pd
import os
import math
<<<<<<< HEAD
=======
import numba
>>>>>>> 34e1d85 (nothing)

from src.common import utils
from src.models import weather

<<<<<<< HEAD
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
=======
@numba.njit
def gaussian_interpolation(target_lat, target_lon, data_lats, data_lons, data_sigmas, data_quantities):
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

    distances = utils.haversine_distance(target_lat, target_lon, data_lats, data_lons)
    # If the target is too far way from any station (5 sigma), the measure is not reliable.
    if min(distances) > 3*np.mean(data_sigmas):
        return np.nan
    weights = np.exp(-distances**2/(2*data_sigmas**2/10))
    weights = weights/np.sum(weights)
    avg = np.sum(weights*data_quantities)
    return avg

class WeatherInterpolator:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

    def log_verbose(self, message):
        """
        Log a message if a logger is configured.

        This method conditionally logs messages based on whether `self.logger`
        is configured. If `self.logger` is not None, the provided message is
        logged. This allows for flexible logging control within classes and functions,
        enabling logging when needed without modifying the internal logging calls.

        Parameters:
        - message (str): The message to be logged.
        """
        if self.logger:
            self.logger.log(message)
>>>>>>> 34e1d85 (nothing)

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

<<<<<<< HEAD
    def estimate_scalar(self, target, scalar):

        relevant_data = self.load_relevant_files(target['time'])
        relevant_data = weather.calibrate_stations(relevant_data)
        relevant_data['f{scalar}_elev'] = [row[f'{scalar}_model'].predict([target['elevation']])[0] for _, row in relevant_data.iterrows()]
        relevant_data = relevant_data[['station', 'lat', 'lon', 'elevation', 'timestamp', f'{scalar}_elev', 'sigma']].copy()
        relevant_data = relevant_data[abs(relevant_data['timestamp'] - target['timestamp']) <= 3600/2]
        relevant_data = relevant_data.dropna(subset=[f'{scalar}_elev'])
        relevant_data = relevant_data.groupby('station').agg({
=======
    def estimate_scalars(self, target, scalars, stations_data=None):

        # self.log_verbose(f'Estimating {scalars} at {target["timestamp"]}')

        if stations_data is None:
            # self.log_verbose('Loading relevant files from database')
            relevant_data = self.load_relevant_files(target['timestamp'])
        else:
            relevant_data = stations_data.copy()

        # self.log_verbose('Cutting data to time threshold')
        time_mask = abs(relevant_data['timestamp'] - target['timestamp']) <= self.config['statistics']['interpolation']['weather']['time-thresh']
        lat_mask = abs(relevant_data['lat'] - target['lat']) <= 1
        lon_mask = abs(relevant_data['lon'] - target['lon']) <= 1
        relevant_data = relevant_data[time_mask & lat_mask & lon_mask]
        # relevant_data = relevant_data[time_mask]
        # self.log_verbose(f'{len(relevant_data)} stations entries in time and position threshold')

        if not any(time_mask & lat_mask & lon_mask):
            return np.repeat(np.nan, len(scalars))
        
        for scalar in scalars:
            # self.log_verbose('Estimating scalar at proper elevation')
            relevant_data[f'{scalar}_h'] = [
                weather.estimate_scalars(
                    target['elevation'],
                    scalar,
                    {
                        'tmpf': row['tmpf'],
                        'elevation': row['elevation'],
                        'clouds': row[['skyc1', 'skyc2', 'skyc3', 'skyc4']].apply(pd.to_numeric, errors='coerce').values,
                        'cloud_levels': row[['skyl1', 'skyl2', 'skyl3', 'skyl4']].apply(pd.to_numeric, errors='coerce').values
                    },
                    self.config)
                for _, row in relevant_data.iterrows()]
        # self.log_verbose('Removing useless columns')
        relevant_data = relevant_data[['station', 'lat', 'lon', 'elevation', 'timestamp', 'sigma'] + [f'{scalar}_h' for scalar in scalars]].copy()
        # self.log_verbose('Removing NaNs')
        for scalar in scalars:
            relevant_data = relevant_data.dropna(subset=[f'{scalar}_h'])
        # self.log_verbose('Grouping by station and averaging')
        agg_dict = {
>>>>>>> 34e1d85 (nothing)
            'lon': 'mean',
            'lat': 'mean',
            'elevation': 'mean',
            'timestamp': 'mean',
<<<<<<< HEAD
            f'{scalar}_elev': 'mean',
            'sigma': 'mean'
        })
        scalar = self.gaussian_interpolation(target, relevant_data, f'{scalar}_elev')

        return scalar
=======
            'sigma': 'mean'
        }
        for scalar in scalars:
            agg_dict[f'{scalar}_h'] = 'mean'
        relevant_data = relevant_data.groupby('station').agg(agg_dict)
        if len(relevant_data) == 0:
            return np.repeat(np.nan, len(scalars))
        # self.log_verbose('Estimating scalar at target location')
        scalars_interpolated = np.zeros(len(scalars))
        for j, scalar in enumerate(scalars):
            #, target_lat, target_lon, data_lats, data_lons, data_sigmas, data_quantities
            #print(target['lat'], target['lon'], relevant_data['lat'].values, relevant_data['lon'].values, relevant_data['sigma'].values, relevant_data[f'{scalar}_h'].values)
            scalars_interpolated[j] = gaussian_interpolation(
                target['lat'],
                target['lon'],
                relevant_data['lat'].values,
                relevant_data['lon'].values,
                relevant_data['sigma'].values,
                relevant_data[f'{scalar}_h'].values)

        return scalars_interpolated
>>>>>>> 34e1d85 (nothing)

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
<<<<<<< HEAD
            temperatures[i] = self.estimate_scalar(target, 'tmpf')
            pressures[i] = self.estimate_scalar(target, 'air_pressure')
            densities[i] = self.estimate_scalar(target, 'air_density')
            clouds[i] = self.estimate_scalar(target, 'clouds')
=======
            temperatures[i] = self.estimate_scalars(target, ['tmpf'])
            pressures[i] = self.estimate_scalars(target, ['air_pressure'])
            densities[i] = self.estimate_scalars(target, ['air_density'])
            clouds[i] = self.estimate_scalars(target, ['clouds'])
>>>>>>> 34e1d85 (nothing)
            
        flight['tmpf'] = temperatures
        flight['air_pressure'] = pressures
        flight['air_density'] = densities
        flight['clouds'] = clouds
            
        flight = self.estimate_flight_wind(flight)

        return flight
