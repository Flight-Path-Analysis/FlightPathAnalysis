"""
This module provides utilities for data parsing and manipulation,
specifically focused on string to number conversion, string results parsing
to pandas DataFrames, date to UNIX timestamp conversion,
and logging functionalities.

Functions:
    - to_number(s): Convert a string input to a number if possible.
    - parse_opensky_to_dataframe(results): Parse the text results of a database query to a pandas DataFrame.
    - to_unix_timestamp(date_input): Convert a given date input to its corresponding UNIX timestamp.

Classes:
    - Logger: Provides logging functionalities with a timestamp. 
              It takes a configuration for its initialization and 
              uses it to format and store log messages.

Dependencies:
    - numpy
    - pandas
    - warnings
    - dateutil.parser
    - datetime
    - yaml
    - os
"""

import os
import datetime
import dateutil.parser

import numpy as np
import pandas as pd
import math
import numba

def to_number(s):
    """
    Convert a given input to a number if possible.
    
    This function attempts to convert the provided input to an integer or 
    floating-point number. If the conversion is not possible, it returns 
    the input as-is.
    
    Parameters:
    - s: The input to be converted. Can be of any type, but typically expected to be a string.
    
    Returns:
    - int or float or same type as input: If `s` can be converted to an
    integer or floating-point number, the converted number is returned.
    Otherwise, the original input is returned.
      
    Examples:
    >>> to_number("45")
    45
    
    >>> to_number("3.14")
    3.14
    
    >>> to_number("hello")
    'hello'
    """
    if not isinstance(s, str):
        return s
    if '.' in s:
        try:
            # Try to convert to a float
            return float(s)
        except ValueError:
            return s
    try:
        # Try to convert to a float
        return int(s)
    except ValueError:
        return s
    
def my_eval(s):
    """
    Converts a string into an integer, float, or date object, depending on the string's content. 
    If the conversion is unsuccessful, returns the original string.

    This function attempts to interpret a string in the following order:
    1. Tries to convert the string to an integer.
    2. If unsuccessful, tries to convert it to a float.
    3. If still unsuccessful, tries to convert it to a date.
    4. If all conversions fail, it returns the original string.

    Parameters:
    - s (str): The string to convert.

    Returns:
    - int, float, datetime.date, or str: The converted value or the original string if all conversions fail.
    """
    try:
        # Try to convert the string to a number (int or float)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            try:
                # Try to convert the string to a datetime date
                return datetime.datetime.strptime(s, '%Y-%m-%d').date()
            except ValueError:
                # If all conversions fail, return the original string
                return s

def parse_opensky_to_dataframe(results):
    """
    Parse the text results of a database query to a pandas DataFrame.
    
    This function processes the provided query results, which is expected to be in a 
    delimited text format. The results are parsed and converted to a pandas DataFrame 
    where each column corresponds to a field in the query results and rows correspond to 
    individual records. If a value in the results can be converted to a number using the 
    `to_number` function, it will be. If an empty string is provided, an empty DataFrame
    is returned.

    Parameters:
    - results (str): A string containing the results of a database query. The results should 
                     be delimited, typically by '|' and organized into rows separated by newlines.
                     
    Returns:
    - pd.DataFrame: A DataFrame representation of the parsed query results.
    
    Warnings:
    - If the input results string is empty, a warning is issued and an empty DataFrame is returned.
    
    Example:
    - Input: \
    +----------+--------+------------+------------+---------------------+-------------------+
     | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
    +----------+--------+------------+------------+---------------------+-------------------+
     | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
    +----------+--------+------------+------------+---------------------+-------------------+
   - Output:
                callsign  icao24   firstseen    lastseen    estdepartureairport  estarrivalairport  
             0  DAL1199   a0a6c5   1688674458   1688676085  KBTR                 NULL
    """
    # Checks if input is a string
    if not isinstance(results, str):
        raise ValueError("Provided results is not in string format.")
    # Checks if input is an empty string
    if not results.strip():
        raise ValueError("Provided results string is empty.")

    lines = results.split('\n')
    if len(lines) < 4:
        raise ValueError(f'Invalid input, results should consist of \
at least 4 lines, even if empty of data.{results}')

    first_line, columns_line, third_line, last_line = lines[0], lines[1], lines[2], lines[-1]

    # Checks if first third, and last lines are properly formatted
    if (len(first_line.replace('+','').replace('-','')) != 0) or \
       (len(third_line.replace('+','').replace('-','')) != 0) or \
       (len(last_line.replace('+','').replace('-','')) != 0):
        raise ValueError("Invalid input, first, third, and last line \
is expected to contain nothing but \"+\" and \"-\"")

    if not columns_line.startswith('|') or \
    not columns_line.endswith('|'):
        raise ValueError(f"Column lines should start with \
\" |\" and end with \"|\". {columns_line}")

    # Extract column names
    columns = columns_line.replace(' ', '').replace('\t', '').split('|')[1:-1]

    # Extract data lines
    data_lines = [line for line in lines[2:-1] if line != columns_line \
                  and not all(char in ['+', '-'] for char in line)]

    # Testing for consistency of data_lines
    n_cols = len(columns)
    for i, line in enumerate(data_lines):
        if line.count('|') != n_cols + 1:
            raise ValueError(f"Invalid input, data line \
{i} does not agree with columns format\n{line}")
        if not line.startswith('|') or not line.endswith('|'):
            raise ValueError(f"Data lines should start with \" |\" and end with \"|\". {i}, {line}")

    # Clean up lines and split them
    data_lines = [line.replace(' ', '').replace('\t', '').split('|')[1:-1] for line in data_lines]

    # Create DataFrame
    if len(data_lines) == 0:
        df_dict = {col: [] for i, col in enumerate(columns)}
    else:
        df_data = np.array(data_lines).T
        df_dict = {col: df_data[i] for i, col in enumerate(columns)}

    df = pd.DataFrame(df_dict).map(to_number)

    return df

def parse_iem_to_dataframe(results):
    """
    Converts a text block of data into a Pandas DataFrame.

    The function processes a long string of data, where each line represents either a comment, 
    a header (with column names), or a row of data. Lines beginning with '#' are comments and 
    are ignored, as are empty lines. The first line of non-commented text is treated as the header, 
    providing the column names for the DataFrame. Subsequent lines of data are split by commas 
    and added to the respective columns.

    Each entry in the data is processed by the 'my_eval' function to attempt conversion into 
    int, float, or date types, falling back to the original string if necessary.

    Parameters:
    - data (str): The block of text data to convert.

    Returns:
    - DataFrame: A Pandas DataFrame representing the structured data.
    """

    # Dictionary that will become a dataframe
    data_values = {}
    # Looping thorugh lines of data
    for line in results.split('\n'):
        # Ignoring empty lines or commented-out lines
        if not line.startswith('#') and len(line) > 0:
            # The first non-commented line contains the data columns.
            # If there's nothing in the dictionary, it means that we're there, redefine the dictionary
            if len(data_values) == 0:
                # Start a dictionary where the keys are the columns, and entries are empty arrays (which we'll populate)
                data_values = {col:[] for col in line.split(',')}
            # If we're not in the columns line, we're in a line that contains data
            else:
                # Split data and append it to each entry of dictionary
                line_data = line.split(',')
                for i, col in enumerate(data_values.keys()):
                    data_values[col] += [my_eval(line_data[i])]
    return pd.DataFrame(data_values)

def to_unix_timestamp(date_input):
    """
    Convert a given date input to its corresponding UNIX timestamp.
    
    This function accepts various date input formats including strings 
    representing dates or UNIX timestamps, date objects, datetime objects, 
    and integer or float representations of UNIX timestamps. It then converts 
    and returns the corresponding UNIX timestamp as an integer.
    It assumes date inputs as being in the UTC timezone
    
    Parameters:
    - date_input: The date input to be converted. Can be a string containing a date in 
                  various formats (e.g., 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS') or a UNIX 
                  timestamp, a date object, a datetime object, or a number (int or float) 
                  representing a UNIX timestamp.
    
    Returns:
    - int: The UNIX timestamp corresponding to the given date input.
    
    Raises:
    - ValueError: If the date input format is unsupported.
    
    Examples:
    >>> to_unix_timestamp("2022-01-01")
    1641016800
    
    >>> to_unix_timestamp("1641016800")
    1641016800
    
    >>> to_unix_timestamp(1641016800)
    1641016800
    
    >>> to_unix_timestamp(datetime.date(2022, 1, 1))
    1641016800
    
    >>> to_unix_timestamp(datetime.datetime(2022, 1, 1, 0, 0))
    1641016800
    """

    # If it's already an integer or float, just convert to integer
    if isinstance(date_input, (int, float, np.int64, np.float64)):
        return int(date_input)

    # If it's a string, try to parse it
    if isinstance(date_input, str):
        try:
            # If it's a string representing a UNIX timestamp
            return int(float(date_input))
        except ValueError:
            try:
                # Parse the string into a datetime object
                dt = dateutil.parser.parse(date_input)
                # Convert the datetime to UTC if it's not already
                if dt.tzinfo is not None:
                    dt = dt.astimezone(datetime.timezone.utc)
                else:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                # Return the UNIX timestamp
                return int(dt.timestamp())
            except Exception as exc:
                raise ValueError("Unsupported date format") from exc

    # If it's a datetime.datetime object
    if isinstance(date_input, datetime.datetime):
        # Convert the datetime to UTC if it's not already
        if date_input.tzinfo is not None:
            date_input = date_input.astimezone(datetime.timezone.utc)
        else:
            date_input = date_input.replace(tzinfo=datetime.timezone.utc)
        return int(date_input.timestamp())

    # If it's a date object, convert to a UTC datetime then to UNIX timestamp
    if isinstance(date_input, datetime.date):
        dt = datetime.datetime(date_input.year,
                               date_input.month,
                               date_input.day,
                               tzinfo=datetime.timezone.utc)
        return int(dt.timestamp())

    raise ValueError("Unsupported date format")

import os

def clean_path(path_str):
    """
    Cleans a file path string by replacing backslashes with forward slashes and removing any empty path elements.

    Args:
        path_str (str): The file path string to clean.

    Returns:
        str: The cleaned file path string.
    """
    path_arr = str(path_str).replace('\\','/').split('/')
    path_arr = [path for path in path_arr if path != '']
    return os.path.join(*path_arr)

class Logger:
    """
    Logger class responsible for handling logging functionalities.

    The Logger uses a provided configuration to format and store
    log messages. The configuration should contain information regarding
    the root directory, logging directory, and the logging tag.
    Log entries are saved with timestamps for easier debugging and traceability.

    Attributes:
    - config (dict): The configuration dictionary used for logging settings.

    Methods:
    - clean_path(path): Ensure the provided path string ends with a '/'.
    - log(text): Log the provided text with a timestamp to the specified log file.
    """
    def __init__(self, config, clear_function=None):
        self.config = config
        self.clf = clear_function

    def log(self, text):
        """
        Log the provided text with a timestamp to the specified log file.

        Parameters:
        - text (str): Text message to be logged.
        """
        if not isinstance(text, str):
            raise ValueError('Text to log must be a string')

        date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        root_dir = clean_path(self.config['base-configs']['root-directory'])
        log_directory = clean_path(self.config['log']['log-directory'])

        # Check if the log directory exists, if not, create it
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        tag = self.config['base-configs']['tag']

        log_file = os.path.join(log_directory,f'{tag}.log')
        with open(log_file, 'a', encoding="utf-8") as f:
            log_entry = f'{date} : {text}\n'
            if self.clf is not None:
                self.clf()
            print(log_entry, end='')  # Printing without extra newline since log_entry includes it
            f.write(log_entry)
def timeout_handler(signum, frame):
    """
    Signal handler for raising a TimeoutError after a timeout.

    This function is intended to be used with the signal module to raise
    a TimeoutError after a certain period of time, interrupting the
    program's flow, which can be caught with a try/except block elsewhere
    in the code. Useful for implementing timeouts on operations that might
    hang or run indefinitely.

    Parameters:
    signum : int
        The signal number being handled. Generally, this will be signal.SIGALRM.
    frame : frame
        The current stack frame at the point the signal occurred. This might be
        used for more advanced handling scenarios, but it's not used in this
        basic handler.

    Raises:
    TimeoutError:
        Always raised to indicate a timeout scenario.
    """
    raise TimeoutError("Operation timed out!")

@numba.njit
def haversine_distance(lat1, lon1, lat2, lon2, R=6.371e6):
    """
    Calculate the Haversine distance between two points on the earth specified by longitude and latitude.

    The Haversine formula calculates the shortest distance over the earth’s surface, giving an 'as-the-crow-flies'
    distance between the points (ignoring any hills, valleys, or other potential obstacles). This function uses
    the radius of the earth specified by `R` but defaults to the mean earth radius if `R` is not provided.

    Args:
    lat1 (float): Latitude of the first point in degrees.
    lon1 (float): Longitude of the first point in degrees.
    lat2 (float): Latitude of the second point in degrees.
    lon2 (float): Longitude of the second point in degrees.
    R (float, optional): Earth radius in meters. Default is the mean earth radius (6.371e6 meters).

    Returns:
    float:  Distance between the points in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = lat1/180*np.pi
    lon1_rad = lon1/180*np.pi
    lat2_rad = lat2/180*np.pi
    lon2_rad = lon2/180*np.pi

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2

    return np.abs(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def haversine_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)

    # Calculate the difference between the longitudes
    dLon = lon2 - lon1

    # Calculate the bearing
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
    initial_bearing = math.atan2(x, y)

    # Convert bearing from radians to degrees
    initial_bearing = math.degrees(initial_bearing)
    
    # Normalize the bearing to be between 0 and 360 degrees
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def meters_to_degrees(meters, latitude, R=6.371e6):
    """
    Convert distance from meters to degrees at a specific latitude.

    Args:
    meters (float): Distance in meters.
    latitude (float): Latitude at which the conversion is happening.

    Returns:
    float: Distance in degrees.
    """
    
    # Conversion from degrees to radians
    latitude_radians = latitude/180*np.pi
    
    # Number of meters in a radian
    meters_per_radian = R

    # Calculate the number of meters per degree of latitude and longitude
    meters_per_degree_lat = (np.pi / 180) * R
    meters_per_degree_lon = (np.pi / 180) * R * np.cos(latitude_radians)

    # Calculate the distance in degrees
    degrees_lat = meters / meters_per_degree_lat
    degrees_lon = meters / meters_per_degree_lon

    # We return the mean value considering changes in latitude may affect the circle shape
    return (degrees_lat + degrees_lon) / 2

def gaussian_interpolation(target, data, quantity):
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

    distances = haversine_distance(target['lat'], target['lon'], data['lat'], data['lon'])
    # If the target is too far way from any station (5 sigma), the measure is not reliable.
    if min(distances) > 3*np.mean(data['sigma']):
        return np.nan
    weights = np.exp(-distances**2/(2*data['sigma']**2/10))
    weights = weights/np.sum(weights)
    avg = np.sum(weights*data[quantity])
    return avg

def format_time(seconds):
    if np.isnan(seconds):
        return 'NaN'
    # Calculate days, hours, minutes, and seconds
    days = int(seconds // (24 * 3600))
    seconds = int(seconds % (24 * 3600))
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    
    # Build the output string
    time_str = ""
    if days > 0:
        time_str += f"{days}d "
    if hours > 0 or days > 0:
        time_str += f"{hours}h "
    if minutes > 0 or hours > 0 or days > 0:
        time_str += f"{minutes}m "
    time_str += f"{seconds}s"

    return time_str
