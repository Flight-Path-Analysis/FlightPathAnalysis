"""
This module provides utilities for data parsing and manipulation,
specifically focused on string to number conversion, string results parsing
to pandas DataFrames, date to UNIX timestamp conversion,
and logging functionalities.

Functions:
    - to_number(s): Convert a string input to a number if possible.
    - parse_to_dataframe(results): Parse the text results of a database query to a pandas DataFrame.
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

def parse_to_dataframe(results):
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
    def __init__(self, config):
        self.config = config

    def clean_path(self, path):
        """
        Initializes the Logger class with the given configuration.

        Parameters:
        - config (dict): The configuration dictionary containing settings
            such as the root directory, logging directory, and the logging tag.
        """
        if isinstance(path, str):
            return path if path.endswith('/') else path + '/'

        raise ValueError('Path must be a string')

    def log(self, text):
        """
        Log the provided text with a timestamp to the specified log file.

        Parameters:
        - text (str): Text message to be logged.
        """
        if not isinstance(text, str):
            raise ValueError('Text to log must be a string')

        date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        root_dir = self.clean_path(self.config['base-configs']['root-directory'])
        log_directory = self.clean_path(self.config['log']['log-directory'])

        log_directory = f'{root_dir}{log_directory}'

        # Check if the log directory exists, if not, create it
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        tag = self.config['base-configs']['tag']

        log_file = f'{log_directory}{tag}.log'
        with open(log_file, 'a', encoding="utf-8") as f:
            log_entry = f'{date} : {text}\n'
            print(log_entry, end='')  # Printing without extra newline since log_entry includes it
            f.write(log_entry)
