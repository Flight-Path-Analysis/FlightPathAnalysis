import numpy as np
import pandas as pd
import warnings
import dateutil.parser
import datetime
import yaml

# Load configurations, path is hardcoded, there might be a better way, but good for now.
with open('../../config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print("Error loading the config file:", exc)

def to_number(s):
    """
    Convert a given input to a number if possible.
    
    This function attempts to convert the provided input to an integer or 
    floating-point number. If the conversion is not possible, it returns 
    the input as-is.
    
    Parameters:
    - s: The input to be converted. Can be of any type, but typically expected to be a string.
    
    Returns:
    - int or float or same type as input: If `s` can be converted to an integer or floating-point number, 
      the converted number is returned. Otherwise, the original input is returned.
      
    Examples:
    >>> to_number("45")
    45
    
    >>> to_number("3.14")
    3.14
    
    >>> to_number("hello")
    'hello'
    """
    
    try:
        # Try to convert to an integer
        return int(s)
    except ValueError:
        try:
            # Try to convert to a floating-point number
            return float(s)
        except ValueError:
            # Not a number
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
    - Input: +----------+--------+------------+------------+---------------------+-------------------+
             | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
             +----------+--------+------------+------------+---------------------+-------------------+
             | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
             +----------+--------+------------+------------+---------------------+-------------------+
   - Output:
                callsign  icao24   firstseen    lastseen    estdepartureairport  estarrivalairport  
             0  DAL1199   a0a6c5   1688674458   1688676085  KBTR                 NULL
    """
    # Checks if input is an empty string
    if not results.strip():
        warnings.warn("Provided results string is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    lines = results.split('\n')
    first_line, columns_line = lines[0], lines[1]

    # Extract data lines and split by delimiter
    data_lines = [
        line.replace(' ', '').replace('\t', '').split('|')[1:-1]
        for line in lines[2:-1] if line != first_line and line != columns_line
    ]
    
    # Extract column names
    columns = columns_line.replace(' ', '').replace('\t', '').split('|')[1:-1]

    # Extract column names
    columns = columns_line.replace(' ', '').replace('\t', '').split('|')[1:-1]

    # Create and return DataFrame
    return pd.DataFrame(np.array(good_lines).T, columns=columns).applymap(to_number)

def to_unix_timestamp(date_input):
    """
    Convert a given date input to its corresponding UNIX timestamp.
    
    This function accepts various date input formats including strings 
    representing dates or UNIX timestamps, date objects, datetime objects, 
    and integer or float representations of UNIX timestamps. It then converts 
    and returns the corresponding UNIX timestamp as an integer.
    
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
    if isinstance(date_input, (int, float)):
        return int(date_input)
    
    # If it's a string, try to parse it
    if isinstance(date_input, str):
        try:
            # If it's a string representing a UNIX timestamp
            return int(float(date_input))
        except ValueError:
            # Otherwise, treat it as a date string and parse it
            return int(dateutil.parser.parse(date_input).timestamp())
    
    # If it's a datetime.datetime object
    if isinstance(date_input, datetime.datetime):
        return int(date_input.timestamp())
    
    # If it's a date object, convert to UNIX timestamp
    if isinstance(date_input, datetime.date):
        return int(datetime.datetime(date_input.year, date_input.month, date_input.day).timestamp())

    raise ValueError("Unsupported date format")

class Logger:
    def __init__(config):
        self.config = config
        
    def clean_path(path):
        """
        Ensure the provided path string ends with a '/'.

        Parameters:
        - path (str): A directory path.

        Returns:
        - str: Path with a trailing '/' if it was missing.
        """
        return path if path.endswith('/') else path + '/'

    def log(text):
        """
        Log the provided text with a timestamp to the specified log file.

        Parameters:
        - text (str): Text message to be logged.

        Note:
        The logging location and filename are derived from the global `config` dictionary.
        """
        date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        log_directory = clean_path(self.config['log']['log-directory'])
        tag = self.config['base-configs']['tag']

        log_file = f'{log_directory}{tag}.log'
        with open(log_file, 'a') as f:
            log_entry = f'{date} : {text}\n'
            print(log_entry, end='')  # Printing without extra newline since log_entry includes it
            f.write(log_entry)