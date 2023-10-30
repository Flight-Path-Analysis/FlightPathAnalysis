import yaml
import numpy as np
import pandas as pd
import os

from src.backend import iem_query
from src.backend import utils
from src.backend import weather_models


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
LOGGER.log('--- Gathering Weather Data ---')
querier = iem_query.Querier(CONFIG, logger=LOGGER)
stations = querier.load_all_station_properties(filter_by_date=True)
database_path = os.path.join(CONFIG['data-gather']['weather']['out-dir'], 'stations_database.csv')
stations.sort_values(by='id', inplace=True)
stations.to_csv(database_path)

start_date = utils.to_unix_timestamp(CONFIG['base-configs']['start-date'])
end_date = utils.to_unix_timestamp(CONFIG['base-configs']['end-date'])
chunk_size = CONFIG['data-gather']['chunk-size']

for date in range(start_date - chunk_size, end_date + chunk_size, chunk_size):
    CONFIG['base-configs']['start-date'] = utils.to_unix_timestamp(date)
    CONFIG['base-configs']['end-date'] = utils.to_unix_timestamp(date + chunk_size)
    # Define filename and filepath based on date
    filename = f'{date}_{date + chunk_size}.csv'
    filepath = os.path.join(CONFIG['data-gather']['weather']['out-dir'], filename)
    if not CONFIG.get('data-gather', {}).get('continue-from-last', True) or not os.path.exists(filepath):
        # query_multiple_station_data
        querier = iem_query.Querier(CONFIG, logger=LOGGER)
        LOGGER.log(f'Gathering weather data for date {date} until date {date + chunk_size}')
        stations_data = querier.query_multiple_station_data(stations)
        stations_data = weather_models.calibrate_stations(stations_data, CONFIG)
        stations_data.to_csv(filepath)
    else:
        LOGGER.log(f'Data for date {date} until date {date + chunk_size} already exists. Skipping...')
LOGGER.log('--- Finished Gathering Weather Data ---')