import numpy as np
import pandas as pd
import os
import math
import numba

from src.common import utils
from src.models import weather

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

    def load_relevant_files(self, time):
        # Determine if the input is a single time or a range
        if isinstance(time, list) and len(time) == 2:
            # Time range provided
            start_time, end_time = time
            time_thresh = self.config['statistics']['interpolation']['weather']['time-thresh']
            start_time -= time_thresh
            end_time += time_thresh
        else:
            # Single time provided, calculate the start and end times of interest
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
            if '_' in file and file.endswith('.csv') and len(file) == 25:
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

    def estimate_scalars(self, target, scalars, stations_data=None):

        # self.log_verbose(f'Estimating {scalars} at {target["time"]}')

        if stations_data is None:
            # self.log_verbose('Loading relevant files from database')
            relevant_data = self.load_relevant_files(target['time'])
        else:
            relevant_data = stations_data.copy()
        # self.log_verbose('Cutting data to time threshold')
        time_mask = abs(relevant_data['time'] - target['time']) <= self.config['statistics']['interpolation']['weather']['time-thresh']
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
                        'sknt': row['sknt'],
                        'clouds': row[['skyc1', 'skyc2', 'skyc3', 'skyc4']].apply(pd.to_numeric, errors='coerce').values,
                        'cloud_levels': row[['skyl1', 'skyl2', 'skyl3', 'skyl4']].apply(pd.to_numeric, errors='coerce').values
                    },
                    self.config)
                for _, row in relevant_data.iterrows()]
        # self.log_verbose('Removing useless columns')
        relevant_data = relevant_data[['station_id', 'lat', 'lon', 'elevation', 'time', 'sigma'] + [f'{scalar}_h' for scalar in scalars]].copy()
        # self.log_verbose('Removing NaNs')
        for scalar in scalars:
            relevant_data = relevant_data.dropna(subset=[f'{scalar}_h'])
        # self.log_verbose('Grouping by station and averaging')
        agg_dict = {
            'lon': 'mean',
            'lat': 'mean',
            'elevation': 'mean',
            'time': 'mean',
            'sigma': 'mean'
        }
        for scalar in scalars:
            agg_dict[f'{scalar}_h'] = 'mean'
        relevant_data = relevant_data.groupby('station_id').agg(agg_dict)
        if len(relevant_data) == 0:
            return np.repeat(np.nan, len(scalars))
        # self.log_verbose('Estimating scalar at target location')
        scalars_interpolated = np.zeros(len(scalars))
        for j, scalar in enumerate(scalars):
            #, target_lat, target_lon, data_lats, data_lons, data_sigmas, data_quantities
            scalars_interpolated[j] = gaussian_interpolation(
                target['lat'],
                target['lon'],
                relevant_data['lat'].values,
                relevant_data['lon'].values,
                relevant_data['sigma'].values,
                relevant_data[f'{scalar}_h'].values)

        return scalars_interpolated

    def estimate_flight_wind(self, flight):
        dx = [utils.haversine_distance(row_a.lat, row_a.lon, row_b.lat, row_b.lon) for row_a, row_b in zip(flight[:-1].itertuples(), flight[1:].itertuples())]
        dt = [row_b.time - row_a.time for row_a, row_b in zip(flight[:-1].itertuples(), flight[1:].itertuples())]
        v = list(np.array(dx)/np.array(dt))

        bearings = [utils.haversine_bearing(row_a.lat, row_a.lon, row_b.lat, row_b.lon) for row_a, row_b in zip(flight[:-1].itertuples(), flight[1:].itertuples())]

        flight['geo_velocity'] = v + [v[-1]]
        flight['geo_heading'] = bearings + [bearings[-1]]

        geo_heading_rad = flight['geo_heading'].apply(math.radians)
        heading_rad = flight['geo_heading'].apply(math.radians)

        # Compoent of the geo velocity in the direction of the heading.
        geo_v_h = flight['geo_velocity']*np.cos(heading_rad - geo_heading_rad)
        # Component of the wind vector in the direction of the heading.
        wind_v_g = geo_v_h - flight['velocity']
        
        flight['wind_velocity_heading'] = wind_v_g*1.94384
        
        return flight

    def compute_flight_weather_quantities(self, scalars, state_vectors, stations_data=None):
        if not isinstance(scalars, list) and not isinstance(scalars, np.ndarray):
            scalars = [scalars]

        if stations_data is None:
            stations_data = self.load_relevant_files([state_vectors.iloc[0]['time'], state_vectors.iloc[-1]['time']])
        scalar_values = {scalar: np.repeat(np.nan, len(state_vectors)) for scalar in scalars}
        step = self.config['statistics']['interpolation']['flights']['step']

        for i, row in state_vectors.iloc[::step].iterrows():
            target = {
                'lon': row['lon'],
                'lat': row['lat'],
                'time': row['time'],
                'elevation': row['geoaltitude'],
                }
            values = self.estimate_scalars(target, scalars, stations_data=stations_data)
            for j, scalar in enumerate(scalars):
                scalar_values[scalar][i] = values[j]
            
        for scalar in scalars:
            state_vectors[scalar] = scalar_values[scalar]
            state_vectors[scalar] = state_vectors[scalar].interpolate(method='linear')

        return state_vectors
    
    def computer_flight_weather_integrals(state_vectors):

        return {'test':0}
