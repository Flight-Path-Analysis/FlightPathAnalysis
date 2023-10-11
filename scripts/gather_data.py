"""
This script is responsible for gathering flight data from the OpenSky database, processing it,
and compressing it using the SplineCompressor.

The main functionalities include:
1. Setting up necessary paths and importing required modules.
2. Loading configuration and credentials from YAML files.
3. Initializing logging and querying tools.
4. Looping through routes-of-interest defined in the configuration, querying for flight data
   between a specified start and end date.
5. For each flight:
    - Querying the associated state vectors from OpenSky.
    - Cleaning and filtering the data.
    - Compressing the data with SplineCompressor and calculating the compression factor.
    - Saving the compressed data as a YAML file.

Dependencies:
    - matplotlib, datetime, yaml, pandas, numpy, and modules from the `src.backend` package.

Configuration:
    - The script reads configuration from a 'config.yml' file, which specifies OpenSky credentials,
      routes-of-interest, and other parameters.
    - Flight data and configuration details are expected to be defined in the `config.yml`.

Exceptions:
    - Raises a ValueError if no OpenSky credentials file is specified in the configuration.
    - Catches and prints exceptions if there's an error while reading the YAML files.

Outputs:
    - The script generates compressed flight data in YAML format and saves it in the
      directory specified in the configuration. The filename corresponds to the flight's unique ID.

Note:
    - Ensure the specified paths, configurations, and dependencies are set up correctly
      before running the script.
"""

import yaml
import numpy as np

from src.backend import opensky_query
from src.backend import utils
from src.backend import compressors

ROOT_PATH = '.'

# Loading config file
with open(f'{ROOT_PATH}/config/config.yml', 'r') as file:
    try:
        CONFIG = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Checking for and loading opensky credentials file.
CREDENTIALS_FILE = CONFIG['base-configs']['opensky-credentials']
if not CREDENTIALS_FILE:
    raise ValueError('No OpenSky credentials file specified in in config.yaml')

with open(f'{ROOT_PATH}/{CREDENTIALS_FILE}', 'r') as file:
    try:
        CREDENTIALS = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Defining Logger
LOGGER = utils.Logger(CONFIG)

# Creates an instane of the Querier class used for querying the opensky database
OPENSKY_QUERIER = opensky_query.Querier(
    CREDENTIALS['username'],
    CREDENTIALS['password'],
    CONFIG['data-gather']['flights']['hostname'],
    CONFIG['data-gather']['flights']['port'],
    logger=LOGGER)

# Creates an instance of the SplineCompressor class.
COMPRESSOR = compressors.SplineCompressor(CONFIG)

# List of columns of state_vectors to be compressed.
COLUMNS_COMPRESS = ['lat', 'lon', 'baroaltitude', 'geoaltitude', 'heading', 'velocity']

for airport_route in CONFIG['data-gather']['flights']['routes-of-interest']:
    start_date = CONFIG['data-gather']['flights']['start-date']
    end_date = CONFIG['data-gather']['flights']['end-date']
    origin_airport = airport_route[0]
    destination_airport = airport_route[1]
    flights = OPENSKY_QUERIER.query_flight_data(
        origin_airport,
        destination_airport,
        start_date,
        end_date)

    for i, flight in flights.iterrows():
        icao24 = flight['icao24']
        firstseen = flight['firstseen']
        lastseen = flight['lastseen']
        estdepartureairport = flight['estdepartureairport']
        estarrivalairport = flight['estarrivalairport']
        flight_id = f"{icao24}_{firstseen}_{lastseen}_\
        {estdepartureairport}_{estarrivalairport}"

        state_vectors = OPENSKY_QUERIER.query_state_vectors(
                        icao24,
                        firstseen,
                        lastseen)
        # Cleaning Data
        cols_to_check = ['time',
                         'lat',
                         'lon',
                         'velocity',
                         'heading',
                         'baroaltitude',
                         'geoaltitude',
                         'hour']
        for col in cols_to_check:
            state_vectors[col] = state_vectors[col].apply(
                lambda x: np.nan if isinstance(x, str) else x)
        state_vectors.dropna(inplace=True)

        cols_to_check = ['lat', 'lon']
        state_vectors = state_vectors.drop_duplicates(subset=cols_to_check, keep='first')

        # Encoding data
        metadata = COMPRESSOR.encode_from_dataframe(state_vectors, 'time', COLUMNS_COMPRESS)
        # Setting flight_id in the metadata
        metadata['flight_id'] = flight_id
        # Computes the compression factor achieved as x_old/x_new.
        compression_factor = COMPRESSOR.compute_compression_factor(
            state_vectors['time'].values,
            state_vectors[COLUMNS_COMPRESS].values,
            metadata)
        print(f"Compression Factor: {compression_factor}")
        # Turns dictionary data into the yaml format
        yaml_data = yaml.dump(metadata, default_flow_style=None)
        # Saves it to a yaml file
        with open(f"{ROOT_PATH}/{CONFIG['data-gathe']['flights']}/{flight_id}.yml", 'w') as f:
            f.write(yaml_data)
