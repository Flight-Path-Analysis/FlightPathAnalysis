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
# import sys
# sys.path.append('.')
import yaml
import numpy as np
import pandas as pd
import os

from src.backend import opensky_query
from src.backend import utils
from src.backend import compressors

ROOT_PATH = '.'

# Loading config file
with open(f'{ROOT_PATH}/config/config.yml', 'r', encoding="utf-8") as file:
    try:
        CONFIG = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Checking for and loading opensky credentials file.
CREDENTIALS_FILE = CONFIG['base-configs']['opensky-credentials']
if not CREDENTIALS_FILE:
    raise ValueError('No OpenSky credentials file specified in in config.yaml')

with open(f'{ROOT_PATH}/{CREDENTIALS_FILE}', 'r', encoding="utf-8") as file:
    try:
        CREDENTIALS = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Defining Logger
LOGGER = utils.Logger(CONFIG)

CREDENTIALS['hostname'] = CONFIG['data-gather']['flights']['hostname']
CREDENTIALS['port'] = CONFIG['data-gather']['flights']['port']
CREDENTIALS['bad_days_csv'] = CONFIG['data-gather']['flights']['bad-days-csv']
CREDENTIALS['chunk_size'] = CONFIG['data-gather']['flights']['chunk-size']
CREDENTIALS['flight_data_timeout'] = CONFIG['data-gather']['flights']['flight-data-timeout']
CREDENTIALS['state_vector_timeout'] = CONFIG['data-gather']['flights']['flight-data-timeout']
CREDENTIALS['flight_data_retries'] = CONFIG['data-gather']['flights']['flight-data-retries']
CREDENTIALS['state_vector_retries'] = CONFIG['data-gather']['flights']['flight-data-retries']

# Creates an instane of the Querier class used for querying the opensky database
OPENSKY_QUERIER = opensky_query.Querier(
    CREDENTIALS,
    logger=LOGGER)

# Creates an instance of the SplineCompressor class.
COMPRESSOR = compressors.CsvCompressor(CONFIG, logger=LOGGER)

# List of columns of state_vectors to be compressed.
COLUMNS_COMPRESS = ['lat', 'lon', 'baroaltitude', 'geoaltitude', 'heading', 'velocity']

for airport_route in CONFIG['data-gather']['flights']['routes-of-interest']:
    start_date = CONFIG['data-gather']['flights']['start-date']
    end_date = CONFIG['data-gather']['flights']['end-date']
    departure_airport = airport_route[0]
    arrival_airport = airport_route[1]
    flights_data_id = f"{departure_airport}_{arrival_airport}_{start_date}_{end_date}"
    directory = f"{ROOT_PATH}/{CONFIG['data-gather']['flights']['out-dir']}/\
{departure_airport}_{arrival_airport}/"
    if not CONFIG['data-gather']['flights']['continue-from-last'] or \
        f'{flights_data_id}.csv' not in os.listdir(directory):
        flights = OPENSKY_QUERIER.query_flight_data(
            {'departure_airport': departure_airport,
            'arrival_airport': arrival_airport},
            {'start': start_date,
            'end': end_date})
        flights.to_csv(f"{directory}/{flights_data_id}.csv")
    else:
        flights = pd.read_csv(f"{directory}/{flights_data_id}.csv", index_col=0)

    for i, flight in flights.iterrows():
        icao24 = flight['icao24']
        firstseen = flight['firstseen']
        lastseen = flight['lastseen']
        estdepartureairport = flight['estdepartureairport']
        estarrivalairport = flight['estarrivalairport']
        flight_id = f"{icao24}_{firstseen}_{lastseen}_\
{estdepartureairport}_{estarrivalairport}"
        filename = f"{directory}/state_vectors/{flight_id}.csv"
        if not CONFIG['data-gather']['flights']['continue-from-last'] or \
            f'{flight_id}.csv' not in os.listdir(
                f"{directory}/state_vectors/"):
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
            if not os.path.exists(f'{directory}/state_vectors/'):
                os.makedirs(f'{directory}/state_vectors/')
            COMPRESSOR.encode_from_dataframe_to_file(
                state_vectors, filename)

print('Done!')