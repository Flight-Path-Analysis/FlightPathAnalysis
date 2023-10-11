import matplotlib.pyplot as plt 
import datetime
import yaml
import pandas as pd
import numpy as np

# Path from this script to the repo's root directory
root_path = '.'
import sys
sys.path.append(root_path)

from src.backend import opensky_query
from src.backend import utils
from src.backend import compressors

# Loading config file
with open(f'{root_path}/config/config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
# Checking for and loading opensky credentials file.
credentials_file = config['base-configs']['opensky-credentials']
if not credentials_file:
    raise ValueError('No OpenSky credentials file specified in in config.yaml')

with open(f'{root_path}/{credentials_file}', 'r') as file:
    try:
        credentials = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
# Defining Logger
logger = utils.Logger(config)

# Creates an instane of the Querier class used for querying the opensky database
opensky_querier = opensky_query.Querier(
    credentials['username'], 
    credentials['password'],
    config['data-gather']['flights']['hostname'], 
    config['data-gather']['flights']['port'], 
    logger = logger)

# Creates an instance of the SplineCompressor class.
compressor = compressors.SplineCompressor(config)

# List of columns of state_vectors to be compressed.
columns_compress = ['lat', 'lon', 'baroaltitude', 'geoaltitude', 'heading', 'velocity']

for airport_route in config['data-gather']['flights']['routes-of-interest']:
    start_date = config['data-gather']['flights']['start-date']
    end_date = config['data-gather']['flights']['end-date']
    origin_airport = airport_route[0]
    destination_airport = airport_route[1]
    flights = opensky_querier.query_flight_data(
          origin_airport, 
          destination_airport, 
          start_date, 
          end_date)
    
    for i, flight in flights.iterrows():
        flight_id = f"{flight['icao24']}_{flight['firstseen']}_{flight['lastseen']}_{flight['estdepartureairport']}_{flight['estarrivalairport']}"
        state_vectors = opensky_querier.query_state_vectors(
                        flight['icao24'],
                        flight['firstseen'],
                        flight['lastseen'])
        # Cleaning Data
        cols_to_check = ['time', 'lat', 'lon', 'velocity', 'heading', 'baroaltitude', 'geoaltitude', 'hour']
        for col in cols_to_check:
            state_vectors[col] = state_vectors[col].apply(lambda x: np.nan if isinstance(x, str) else x)
        state_vectors.dropna(inplace=True)

        cols_to_check = ['lat', 'lon']
        state_vectors = state_vectors.drop_duplicates(subset=cols_to_check, keep='first')
        
        # Encoding data
        metadata = compressor.encode_from_dataframe(state_vectors, 'time', columns)
        # Setting flight_id in the metadata
        metadata['flight_id'] = flight_id
        # Computes the compression factor achieved as x_old/x_new.
        compression_factor = compressor.compute_compression_factor(state_vectors['time'].values, state_vectors[columns].values, metadata)
        print(f"Compression Factor: {compression_factor}")
        # Turns dictionary data into the yaml format
        yaml_data = yaml.dump(metadata, default_flow_style=None)
        # Saves it to a yaml file
        with open(f"{root_path}/{config['data-gathe']['flights']}/{flight_id}.yml", 'w') as f:
            f.write(yaml_data)
        

    
