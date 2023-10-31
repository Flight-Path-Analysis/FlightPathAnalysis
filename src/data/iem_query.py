import pandas as pd
import signal
import datetime
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import json
import time
import numpy as np

from src.backend import utils

class Querier:
    def __init__(self, config, logger=None):

        self.config = config
        self.logger = logger

    # Function that will log text if logger is different than None
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
    
    def load_all_station_properties(self, filter_by_date=False, compute_sigma=True):
        """
        Loads all station properties from the IEM ASOS network and returns them as a pandas DataFrame.

        Args:
            filter_by_date (bool, optional): Whether to filter the stations by date. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the properties of all stations in the IEM ASOS network.
        """
        timeout = self.config['data-gather']['timeout']
        retries = self.config['data-gather']['retries']
        # Array containing all the US States 2-Letter codes
        states = """AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME
        MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
        WA WI WV WY"""
        # Array to hold the dictionary correspondent to each station
        stations = []
        # Setting up the timeout signal
        signal.signal(signal.SIGALRM, utils.timeout_handler)
        # Looping through states
        for state in states.split():
            attempt = 0
            success = False
            while attempt < retries and not success:
                try:
                    self.log_verbose(f'Loading stations for state {state}')

                    # Start the timer. Once the time is out, a TimeoutError is raised
                    signal.alarm(timeout)
                    # Manipulating the uri to the proper network corresponding to the state
                    uri = f"https://mesonet.agron.iastate.edu/geojson/network/{state}_ASOS.geojson"
                    # Loading the data from the uri of the state
                    data = urlopen(uri)
                    # Appending the list of stations gathered, interpreted by json
                    stations += json.load(data)["features"]
                    # Reset the alarm
                    signal.alarm(0)
                    success = True
                except TimeoutError:
                    attempt += 1
                    self.log_verbose(f"Timeout. Retrying {attempt}/{retries}...")
                    time.sleep(10)
                except (URLError, HTTPError) as e:
                    attempt += 1
                    self.log_verbose(f"Error occurred: {e}. Retrying {attempt}/{retries}...")
                    time.sleep(10)
                except json.JSONDecodeError as e:
                    attempt += 1
                    self.log_verbose(f"JSON decoding error: {e}. Retrying {attempt}/{retries}...")
                    time.sleep(10)
                except Exception as e:
                    attempt += 1
                    self.log_verbose(f"An unexpected error occurred: {e}. Retrying {attempt}/{retries}...")
                    time.sleep(10)
        # Cleanup dictionaries here (as per your existing logic)
        self.log_verbose("Cleaning Up Station Information")
        for station in stations:
            self.log_verbose(f"Cleaning Up Station Information for station {station['id']}")
            for key in station['properties'].keys():
                station[key] = station['properties'][key]
            del station['properties']
            station['lon'] = station['geometry']['coordinates'][0]
            station['lat'] = station['geometry']['coordinates'][1]
            del station['geometry']
            del station['type']
            del station['sid']
        if filter_by_date:
            self.log_verbose("Filtering stations by date")
            stations = self.filter_stations_by_date(stations)

        stations = pd.DataFrame(stations)

        if compute_sigma:
            self.log_verbose("Computing stations sigma")
            stations = self.compute_stations_sigma(stations)

        return stations
    
    def compute_stations_sigma(self, stations):
        n_avg_sigma = self.config['data-gather']['weather']['n-avg-sigma']
        sigmas = np.zeros(len(stations))
        for i, row in stations.iterrows():
            # For every station, compute the distance to all other locations
            distances = utils.haversine_distance(row['lat'], row['lon'], stations['lat'], stations['lon'])
            # Sort them from smallest to largers
            distances = sorted(distances)
            # Take the average of the closest 5 (not including itself)
            sigmas[i] = np.mean(distances[1:n_avg_sigma+1])
        stations['sigma'] = sigmas
        return stations
    
    # Function to filter stations to ones that have any data in between startts and endts
    def filter_stations_by_date(self, stations):
        """
        Filter the list of station dictionaries. Include a station if its operation 
        dates overlap with the provided startts and endts, even partially.

        Parameters:
        - stations (list of dicts): The list of station dictionaries.
        - startts (datetime): The start datetime.
        - endts (datetime): The end datetime.

        Returns:
        - list of dicts: The filtered list of stations.
        """
        startts = utils.to_unix_timestamp(self.config['base-configs']['start-date'])
        endts = utils.to_unix_timestamp(self.config['base-configs']['end-date'])
        startts = datetime.datetime.fromtimestamp(startts)
        endts = datetime.datetime.fromtimestamp(endts)

        filtered_stations = []

        for station in stations:
            # Parse the station's operation start and end dates.
            # It's important to handle cases where these dates are None or empty strings.
            archive_begin_str = station.get('archive_begin', '')
            archive_end_str = station.get('archive_end', '')

            try:
                archive_begin = datetime.datetime.strptime(archive_begin_str, '%Y-%m-%d') if archive_begin_str else None
            except ValueError:
                archive_begin = None

            try:
                # If archive_end is None or an empty string, we assume the station is still operational
                archive_end = datetime.datetime.strptime(archive_end_str, '%Y-%m-%d') if archive_end_str else datetime.datetime.max
            except ValueError:
                archive_end = None

            # Check if the station's operation dates overlap with the query dates.
            # The station is considered valid if its operation dates overlap even partially with the query dates.
            if archive_begin is not None and archive_end is not None:
                if (startts <= archive_end and endts >= archive_begin):
                    filtered_stations.append(station)

        return filtered_stations

    def clean_iem_data(self, data):
        """
        Clean the raw IEM meteorological data for consistency and ease of use.

        This method converts specific meteorological data points into more universally applicable units, handles missing data, and sorts records. For example, it converts temperature readings from Fahrenheit to Celsius, wind speeds from knots to meters per second, and wind direction to vector components. It ensures the data is suitable for analysis or other operations.

        Parameters:
        data (pd.DataFrame): A DataFrame containing raw weather data as obtained from the IEM network.

        Returns:
        pd.DataFrame: The cleaned and restructured meteorological data ready for analysis.

        Note:
        The cleaning operations are specific to the IEM data structure. If the data source changes, this method may require adjustments to handle different units or data representations.
        """

        # Convert the 'valid' column into Unix timestamps, and then to datetime objects
        data['timestamp'] = data['valid'].apply(utils.to_unix_timestamp)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

        # Replace 'M' (indicating missing data) with NaN for proper handling in pandas operations
        data = data.replace('M', np.nan)

        # Convert wind speed from knots to meters per second
        data['smps'] = data['sknt']*0.514444

        # Calculate the eastward and northward components of the wind speed in knots
        data['sknt_E'] = np.sin(data['drct']/180*np.pi)*data['sknt']
        data['sknt_N'] = np.cos(data['drct']/180*np.pi)*data['sknt']

        # Calculate the eastward and northward components of the wind speed in meters per second
        data['smps_E'] = np.sin(data['drct']/180*np.pi)*data['smps']
        data['smps_N'] = np.cos(data['drct']/180*np.pi)*data['smps']

        # Convert temperature from Fahrenheit to Celsius
        data['tmpc'] = (data['tmpf']-32)*5/9

        # Sort the data by timestamp, reindexing the DataFrame
        data.sort_values(by='timestamp', ignore_index=True, inplace=True)

        data = self.interprate_metar(data)
        # Return the cleaned DataFrame
        return data
    
    def interprate_metar(self, data):
        # Eventually here
        return data
        
    def interpolate_iem_data(self, data):
        """
        Interpolate missing values in a DataFrame containing weather data.

        This method interpolates missing numeric values in the DataFrame, based on time, ensuring a continuous dataset for better accuracy in subsequent analysis. Non-numeric columns are not interpolated but are retained. The DataFrame must contain data from only one weather station, and the 'timestamp' column must be present for time-based interpolation.

        Parameters:
        data (pd.DataFrame): A DataFrame containing weather data from a single station. Each row corresponds to a timestamped record of various weather parameters. It is imperative that the DataFrame has a 'timestamp' column, which will be used as the index during the interpolation process.

        Raises:
        ValueError: If the input DataFrame contains data from multiple stations, as indicated by the 'station' column.
        ValueError: If the 'timestamp' column is missing in the input DataFrame, which is essential for time-based interpolation.

        Returns:
        pd.DataFrame: A DataFrame with missing values in numeric columns interpolated based on time. The original structure (columns and indices) of the DataFrame is maintained, with 'timestamp' no longer set as the index, reflecting its state upon input.

        Notes:
        The method is designed to handle numeric data types for interpolation. Columns that do not contain numeric data are excluded from the interpolation process but are added back to the final DataFrame. Therefore, the returned DataFrame maintains the original structure and data types, with missing values in numeric columns filled.

        The interpolation is 'time' based, assuming linear changes between consecutive timestamps. It is crucial for the 'timestamp' column to be accurate, as interpolation relies on the temporal distance between records.
        """
        if len(np.unique(data['station'])) > 1:
            raise ValueError('Dataframe must contain data from only one station!')
        if 'timestamp' not in data.columns:
            raise ValueError('Dataframe must be cleaned before interpolation!')

        # Set the timestamp as the index
        data.set_index('timestamp', inplace=True)

        # Ensure the data frame only contains columns with numeric data
        # This is done by selecting only the columns that have numeric data types
        numeric_data_cols = data.select_dtypes(include=[np.number])

        # Perform interpolation only on the numeric columns
        interpolated_data = numeric_data_cols.interpolate(method='time', limit_direction='both')

        # Now, you may want to merge back non-numeric columns which were excluded from interpolation
        for col in data.columns:
            if col not in interpolated_data.columns:
                interpolated_data[col] = data[col]

        # Reset the index after interpolation
        interpolated_data.reset_index(inplace=True)

        interpolated_data['timestamp'] = interpolated_data['timestamp'].apply(utils.to_unix_timestamp)

        return interpolated_data

  
    def query_station_data(self, station, clean_data=True, interpolate_data=True, pass_station_data=True):
        """
        Query, process, and interpolate meteorological data for a specific station within a given time frame.

        This method retrieves weather data from the specified station, performs cleaning, filters columns if necessary, and applies time-based interpolation to numeric data. The method works within the time frame specified in the configuration, ensuring the data is relevant and consistent with the user's requirements.

        Parameters:
        station (dict): A dictionary containing the station's information, including at least the 'id' key corresponding to the station's identifier.
        filter_columns (bool, optional): If True, filters the DataFrame to only include columns listed in 'columns-of-interest' from the configuration. Defaults to False.
        clean_data (bool, optional): If True, performs cleaning operations on the data, such as converting temperatures to Celsius, wind speeds to meters per second, and replacing 'M' (missing data) with NaN. Defaults to True.
        interpolate_data (bool, optional): If True, applies time-based interpolation to fill in missing values in the DataFrame, ensuring a continuous dataset. Defaults to True.

        Returns:
        pd.DataFrame: A DataFrame containing the processed meteorological data from the specified station.

        Raises:
        URLError: If there is a problem with the connection to the data service.

        Note:
        This method relies on several other methods within the class, such as 'clean_iem_data' for cleaning and 'interpolate_iem_data' for interpolation. Ensure these methods are properly defined and tested for accurate results.
        """

        # Convert configured start and end dates to Unix timestamp and then to datetime objects for manipulation
        startts = utils.to_unix_timestamp(self.config['base-configs']['start-date'])
        endts = utils.to_unix_timestamp(self.config['base-configs']['end-date'])
        startts = datetime.datetime.fromtimestamp(startts)
        endts = datetime.datetime.fromtimestamp(endts)

        # Retrieve the columns of interest from the configuration file
        columns_of_interest = self.config['data-gather']['weather']['columns-of-interest']

        # Construct the base URL for the data service
        service = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?data=all&tz=Etc/UTC&format=comma&latlon=yes&"

        # Manipulate the URL to include the date parameters formatted properly
        service += startts.strftime("year1=%Y&month1=%m&day1=%d&") + endts.strftime("year2=%Y&month2=%m&day2=%d&")

        # Construct the complete URI including the station ID
        uri = f"{service}&station={station['id']}"

        # Log the constructed URL for verbose output and debugging
        self.log_verbose(f"Requesting data for station {station['id']} on base-url: {service}")

        # Attempt to retrieve the data from the constructed URL
        raw_data = urlopen(uri, timeout=300).read().decode("utf-8")

        # Parse the raw CSV data into a pandas DataFrame
        data = utils.parse_iem_to_dataframe(raw_data)

        # If filter_columns is True, keep only the columns of interest in the DataFrame
        if columns_of_interest is not None:
            data = data[columns_of_interest]

        # If clean_data is True, apply the cleaning method to the DataFrame
        if clean_data:
            data = self.clean_iem_data(data)

        if pass_station_data:
            # Assigning elevation to each measurement
            data['elevation'] = station['elevation']
            if 'sigma' in station.keys():
                data['sigma'] = station['sigma']
        # If interpolate_data is True, apply the interpolation method to the DataFrame
        if interpolate_data:
            data = self.interpolate_iem_data(data)
        data['station'] = station['id']
        # Return the final, processed DataFrame
        return data

    def query_multiple_station_data(self, stations, clean_data=True, interpolate_data=True, pass_station_data=True):
        """
        Retrieve, process, and interpolate meteorological data for multiple stations.

        This method iteratively accesses weather data from multiple stations provided in a DataFrame. For each station, it retrieves the data, optionally performs cleaning and filtering, and applies interpolation if required. All the station-specific data is then concatenated into a single DataFrame.

        Parameters:
        stations (pd.DataFrame): A DataFrame where each row contains information about a weather station, primarily its 'id'.
        filter_columns (bool, optional): If True, the method retains only the columns specified in the configuration's 'columns-of-interest'. Defaults to False.
        clean_data (bool, optional): Determines whether to clean data points (e.g., unit conversions, handling missing values). Defaults to True.
        interpolate_data (bool, optional): If True, the method performs time-based interpolation on numeric columns to fill in missing data. Defaults to True.

        Returns:
        pd.DataFrame: A DataFrame consolidating the processed data from all specified weather stations.

        Raises:
        ValueError: If the 'stations' DataFrame is empty or the individual station data retrieval fails.

        Note:
        The method depends on 'query_station_data' for data retrieval per station. Ensure this dependency is maintained and updated in line with any changes to the 'query_station_data' method specifications or functionality.
        """

        if stations.empty:
            raise ValueError("No stations provided for data retrieval.")

        # Initialize a list to store DataFrame objects for each station.
        station_data_frames = []

        # Iterating through each row in stations DataFrame
        for _, station in stations.iterrows():
            try:
                # Retrieve data for the current station, with specified parameters for filtering, cleaning, and interpolation.
                station_data = self.query_station_data(station.to_dict(),
                                                    clean_data=clean_data,
                                                    interpolate_data=interpolate_data,
                                                    pass_station_data=pass_station_data)
                # Add the station's data to the list of DataFrames.
                station_data_frames.append(station_data)
            except Exception as e:
                # Handle exceptions raised during data retrieval for individual stations to ensure the process continues with other stations.
                self.log_verbose(f"Data retrieval failed for station {station.get('id', 'Unknown')}: {str(e)}")

        # Concatenate all individual station DataFrames into a single DataFrame.
        # This is more efficient after all data retrievals are complete rather than during the iterative process.
        if station_data_frames:
            all_stations_data = pd.concat(station_data_frames, ignore_index=True)
        else:
            raise ValueError("Data retrieval failed for all stations.")

        return all_stations_data
