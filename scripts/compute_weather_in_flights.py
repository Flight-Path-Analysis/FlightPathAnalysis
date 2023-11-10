import yaml
import numpy as np
import pandas as pd
import os

from src.common import utils
from src.models import weather_models
from src.data import compressors
from src.analysis import weather_interpolator

ROOT_PATH = '.'

# Loading config file
config_path = os.path.join(ROOT_PATH, 'config', 'config.yml')
with open(config_path, 'r', encoding="utf-8") as file:
    try:
        CONFIG = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
CONFIG['data-gather']['weather']['out-dir'] = utils.clean_path(CONFIG['data-gather']['weather']['out-dir'])
CONFIG['data-gather']['flights']['out-dir'] = utils.clean_path(CONFIG['data-gather']['flights']['out-dir'])
CONFIG['log']['log-directory'] = utils.clean_path(CONFIG['log']['log-directory'])

# Defining Logger
LOGGER = utils.Logger(CONFIG)

COMPRESSOR = compressors.CsvCompressor(CONFIG, logger=LOGGER)

INTERPOLATOR = weather_interpolator.WeatherInterpolator(CONFIG, logger=LOGGER)

LOGGER.log('--- Computing Weather Quantities for Flights ---')

flight_paths = CONFIG['data-gather']['routes-of-interest']

scalars = ['tmpf', 'air_pressure', 'air_density', 'clouds']

for flight_path in flight_paths:
    LOGGER.log(f'Computing weather quantities for flights from {flight_path[0]} to {flight_path[1]}')

    start_date = CONFIG['base-configs']['start-date']
    end_date = CONFIG['base-configs']['end-date']
    departure_airport = flight_path[0]
    arrival_airport = flight_path[1]
    flights_data_id = f"{departure_airport}_{arrival_airport}_{start_date}_{end_date}"

    directory_name = f'{flight_path[0]}_{flight_path[1]}'
    flight_data = pd.read_csv(os.path.join(ROOT_PATH, CONFIG['data-gather']['flights']['out-dir'], directory_name, f'{flights_data_id}.csv'))
    flight_data['departure_airport'] = flight_data['estdepartureairport']
    flight_data['arrival_airport'] = flight_data['estarrivalairport']
    flight_data['flight_id'] = [f'{row["icao24"]}_{row["firstseen"]}_{row["lastseen"]}_{row["departure_airport"]}_{row["arrival_airport"]}' for _, row in flight_data.iterrows()]
    
    data_directory = os.path.join(ROOT_PATH, CONFIG['data-gather']['flights']['out-dir'], directory_name, 'state_vectors')
    
    flight_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    flight_ids = [f[:-4] for f in flight_files]
    valid_mask = flight_data['flight_id'].apply(lambda x: x in flight_ids)
    flight_data = flight_data[valid_mask]
    
    flight_data.set_index('flight_id', inplace=True)

    for file in flight_files:
        LOGGER.log(f'Loading file {file}')
        flight = COMPRESSOR.decode_to_dataframe_from_file(os.path.join(data_directory, file))
        LOGGER.log(f'Computing weather quantities for {file}')
        flight = INTERPOLATOR.compute_flight_weather_quantities(scalars, flight)
        integrals = INTERPOLATOR.computer_flight_weather_integrals(flight)
        for key, value in integrals.items():
            flight_data.loc[file[:-4], key] = value
        
        LOGGER.log(f'Saving file {file}')
        COMPRESSOR.encode_from_dataframe_to_file(flight, os.path.join(data_directory, file))

LOGGER.log('--- Finished Computing Weather Quantities for Flights ---')