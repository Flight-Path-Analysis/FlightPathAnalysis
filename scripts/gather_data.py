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
config_path = os.path.join(ROOT_PATH,'config','config.yml')
with open(config_path, 'r', encoding="utf-8") as file:
    try:
        CONFIG = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

CONFIG['data-gather']['flights']['out-dir'] = utils.clean_path(CONFIG['data-gather']['flights']['out-dir'])
CONFIG['data-gather']['flights']['bad-days-csv'] = utils.clean_path(CONFIG['data-gather']['flights']['bad-days-csv'])
CONFIG['base-configs']['opensky-credentials'] = utils.clean_path(CONFIG['base-configs']['opensky-credentials'])
CONFIG['log']['log-directory'] = utils.clean_path(CONFIG['log']['log-directory'])
CONFIG['data-gather']['flights']['airport-list-csv'] = utils.clean_path(CONFIG['data-gather']['flights']['airport-list-csv'])

# Checking for and loading opensky credentials file.
CREDENTIALS_FILE = utils.clean_path(CONFIG['base-configs']['opensky-credentials'])
if not CREDENTIALS_FILE:
    raise ValueError('No OpenSky credentials file specified in in config.yaml')

credentials_path = os.path.join(ROOT_PATH, CREDENTIALS_FILE)
with open(credentials_path, 'r', encoding="utf-8") as file:
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
    airports_df = pd.read_csv(CONFIG['data-gather']['flights']['airport-list-csv'], index_col=0)
    airports_df.set_index('ICAO', inplace=True)
    flights_data_id = f"{departure_airport}_{arrival_airport}_{start_date}_{end_date}"
    directory = os.path.join(ROOT_PATH,
                             CONFIG['data-gather']['flights']['out-dir'],
                             f'{departure_airport}_{arrival_airport}')
    csv_path = os.path.join(directory, f'{flights_data_id}.csv')
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(os.path.join(directory, 'state_vectors')):
                os.makedirs(os.path.join(directory, 'state_vectors'))
    if not CONFIG['data-gather']['flights']['continue-from-last'] or \
        f'{flights_data_id}.csv' not in os.listdir(directory):
        flights = OPENSKY_QUERIER.query_flight_data(
            {'departure_airport': departure_airport,
            'arrival_airport': arrival_airport},
            {'start': start_date,
            'end': end_date})
        flights['attempts'] = [0]*len(flights)
        flights['loaded'] = [False]*len(flights)
        flights.index = range(len(flights))
        flights.to_csv(csv_path)
    else:
        flights = pd.read_csv(csv_path, index_col=0)

    while len(flights[(flights['loaded'] == False) & (flights['attempts'] < 3)]) > 0:
        for i, flight in flights[(flights['loaded'] == False) & (flights['attempts'] < 3)].iterrows():
            icao24 = flight['icao24']
            firstseen = flight['firstseen']
            lastseen = flight['lastseen']
            estdepartureairport = flight['estdepartureairport']
            estarrivalairport = flight['estarrivalairport']
            flight_id = f"{icao24}_{firstseen}_{lastseen}_\
{estdepartureairport}_{estarrivalairport}"
            filename = os.path.join(directory, 'state_vectors', f'{flight_id}.csv')

            flights.at[i, 'attempts'] += 1
            csv_path = os.path.join(directory, f'{flights_data_id}.csv')
            flights.to_csv(csv_path)
            try:
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
                starting_lat_diff = abs(state_vectors.iloc[0]['lat'] - airports_df.loc[estdepartureairport]['lat'])
                ending_lat_diff = abs(state_vectors.iloc[-1]['lat'] - airports_df.loc[estarrivalairport]['lat'])
                starting_lon_diff = abs(state_vectors.iloc[0]['lon'] - airports_df.loc[estdepartureairport]['lon'])
                ending_lon_diff = abs(state_vectors.iloc[-1]['lon'] - airports_df.loc[estarrivalairport]['lon'])

                diffs = [starting_lat_diff, ending_lat_diff, starting_lon_diff, ending_lon_diff ]

                if all(x < CONFIG['data-gather']['flights']['coordinate-distance-thresh'] for x in diffs):
                    # Encoding data
                    state_vector_path = os.path.join(directory, 'state_vectors')
                    if not os.path.exists(state_vector_path):
                        os.makedirs(state_vector_path)
                    COMPRESSOR.encode_from_dataframe_to_file(
                        state_vectors, filename)
                    LOGGER.log(f"Flight {flight_id} loaded successfully.")
                else:
                    LOGGER.log(f"Flight {flight_id} is too far from the airport, skipping.")
                flights.at[i, 'loaded'] = True
                flights.to_csv(csv_path)
            except KeyboardInterrupt:
                LOGGER.log("KeyboardInterrupt caught. Exiting the program.")
                raise
            except:
                LOGGER.log("Failed to load flight, saved for later, skipping for now.")
                pass

LOGGER.log('Done!')