"""
This module provides utilities for querying the OpenSky database via SSH
to retrieve flight data and state vectors.

Dependencies:
- `paramiko`: For managing SSH connections.
- `pandas`: For managing data in a DataFrame format.
- `sys`: For system-specific operations like path adjustments.
- `datetime`: For handling date and time data.
- `utils` from `src.backend`: Local module for utility functions.

Classes:
- Querier: A class that facilitates querying the OpenSky database using SSH.
It provides methods to generate SQL query commands, execute these commands,
and return the results as pandas DataFrames.

Usage:
To use the `Querier` class, an SSH connection needs to be set up.
The hostname, port, username, and password are requiredfor the initialization of
a `Querier` object. Once initialized, the `query_flight_data` and
`query_state_vectors` methods can be used to retrieve data from the 
OpenSky database.
"""
import os
import signal
import datetime
import paramiko
import pandas as pd

from src.backend import utils

def handler(signum, frame):
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

class Querier:
    """
    A class to facilitate querying of the OpenSky database using SSH.

    Attributes:
    - __username (str): Private attribute to store the username.
    - __password (str): Private attribute to store the password.
    - hostname (str): Hostname for the SSH connection.
    - port (int): Port number for the SSH connection.
    - client (paramiko.SSHClient): An SSH client instance to manage SSH connections.
    - logger (utils.Logger): The Logger class instance to log text outputs

    Methods:
    - create_query_command_for_flight_data: Generates the SQL query for flight data.
    - create_query_command_for_state_vectors: Generates the SQL query for state vectors.
    - query_flight_data: Executes the query and returns the results as a DataFrame.
    - query_state_vectors: Executes the query and returns the results as a DataFrame.
    """

    def __init__(self, credentials, logger=None):
        """
        Initialize an instance of the Querier class.

        Parameters:
        - username (str): SSH username.
        - password (str): SSH password.
        - hostname (str): SSH hostname.
        - port (int): SSH port number.
        - client (paramiko.SSHClient): An SSH client instance to manage SSH connections.
        - logger (utils.Logger): The Logger class instance to log text outputs
        """
        self.credentials = credentials
        self.client = paramiko.SSHClient()

        if 'bad_days_csv' in credentials:
            self.bad_days_csv = credentials['bad_days_csv']
        else:
            self.bad_days_csv = None

        self.logger = logger

        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

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

    def create_query_command_for_flight_data(
        self,
        airports,
        dates,
        bad_days,
        limit=None):
        """
        Generate the SQL query command for querying flight data from the OpenSky database.

        Parameters:
        - departure_airport (str): ICAO code of the departure airport.
        - arrival_airport (str): ICAO code of the arrival airport.
        - start_date_unix (int): Start date in UNIX timestamp format.
        - end_date_unix (int): End date in UNIX timestamp format.
        - bad_days (list): List of UNIX timestamps for days to exclude from the query.
        - limit (int, optional): Maximum number of records to retrieve.

        Returns:
        - str: SQL query command.
        """

        departure_airport = airports['departure_airport']
        arrival_airport = airports['arrival_airport']
        start_date_unix = dates['start']
        end_date_unix = dates['end']
        # Begin SQL command
        query = f"""SELECT firstseen, lastseen, callsign, icao24, \
estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = '{departure_airport}' 
    AND estarrivalairport = '{arrival_airport}'
    AND day >= {start_date_unix}
    AND day <= {end_date_unix}
    """
        # Add conditions to exclude bad days
        for day in sorted(bad_days):
            if start_date_unix <= day <= end_date_unix:
                query += f"AND day != {int(day)}\n"

        # Add ordering and limit if specified
        query += "ORDER BY firstseen"
        if not limit:
            query += ";"
        else:
            query += f"\nLIMIT {limit};"
        return query

    def create_query_command_for_state_vectors(
        self, icao24, times, bad_hours, limit=None):
        """
        Generate the SQL query command for querying state vectors data from the OpenSky database.

        Parameters:
        - icao24 (str): ICAO 24-bit address of the aircraft.
        - start_time_unix (int): Start time in UNIX timestamp format.
        - end_time_unix (int): End time in UNIX timestamp format.
        - bad_hours (list): List of UNIX timestamps for hours to exclude from the query.
        - limit (int, optional): Maximum number of records to retrieve.

        Returns:
        - str: SQL query command.
        """
        start_time_unix = times['start']
        end_time_unix = times['end']
        query = f"""SELECT time, lat, lon, velocity, heading, \
baroaltitude, geoaltitude, onground, hour
    FROM state_vectors_data4
    WHERE icao24 = '{icao24}' 
    AND (time >= {start_time_unix} AND time <= {end_time_unix})
    AND (hour > {start_time_unix - 3600} AND hour < {end_time_unix + 3600})
    """
        # Add conditions to exclude bad hours
        for hour in bad_hours:
            query += f"AND hour != {int(hour)}\n"
        if limit:
            query += f"LIMIT {limit}\n"
        # Add ordering and limit if specified
        query += "ORDER BY time;"
        return query

    def execute_query(self, query):
        """
        Execute an SQL query on the OpenSky database and return the results and errors if any.

        Parameters:
        - query (str): The SQL query string to be executed.

        Returns:
        - dict: A dictionary containing:
            - stdout (str): The standard output of the query, typically the query results.
            - stderr (str): The standard error of the query, typically error messages.
        """

        _, stdout, stderr = self.client.exec_command(f"-q {query}")
        return {
            'stdout': stdout.read().decode(),
            'stderr': stderr.read().decode()
        }

    def compute_date_intervals(self, dates_unix):
        """
        Compute date intervals based on the specified Unix start and end dates.

        Parameters:
        - dates_unix (dict): A dictionary containing the 'start' and 'end' Unix timestamps
                              of the desired date range.

        Returns:
        - list: A list of Unix timestamps representing the intervals at which queries will be made.
                The length of each interval is determined by `self.credentials['chunk_size']`.
                The last element in the list is always the 'end' timestamp from `dates_unix`.

        Note:
        The returned list of timestamps represents the intervals that the data fetching
        will occur. If the total time range is less than or equal to 
        `self.credentials['chunk_size']`, the function returns the original start and
        end timestamps. Otherwise, the function returns evenly spaced timestamps at
        intervals of `self.credentials['chunk_size']` between the original start and 
        end timestamps.
        """
        if dates_unix['end'] - dates_unix['start'] <= self.credentials['chunk_size']:
            dates = [dates_unix['start'], dates_unix['end']]
        else:
            dates = list(range(dates_unix['start'], dates_unix['end'],
                               self.credentials['chunk_size']))
            if dates[-1] != dates_unix['end']:
                dates += [dates_unix['end']]
        return dates

    def initialize_bad_days_df(self):
        """
        Initialize a DataFrame to keep track of days for which query attempts 
        have resulted in errors.

        Two scenarios are considered:
        - If `self.bad_days_csv` exists and points to a valid path, the function 
        reads it into a DataFrame. Only the entries from the last week are retained.
        - If `self.bad_days_csv` doesn't point to a valid path, an empty 
        DataFrame is created with columns 'day' and 'date_registered'.

        The resulting DataFrame is then deduplicated based on the 'day' column.

        Returns:
        - pd.DataFrame: A DataFrame with columns 'day' and 'date_registered'
        to store bad days data. Each row contains an integer Unix timestamp under
        'day' and the timestamp when it was registered under 'date_registered'.

        Note:
        'day': Refers to Unix timestamps of days when an error was encountered 
        during data retrieval.
        'date_registered': The datetime object representing when the 
        error day was logged.
        """
        # Array to save the days that return an error from the query
        if self.bad_days_csv and os.path.exists(self.bad_days_csv):
            bad_days_df = pd.read_csv(self.bad_days_csv, index_col = 0,
                                      parse_dates=['date_registered'])

            bad_days_df = bad_days_df[bad_days_df['date_registered'].apply(
                lambda x: (datetime.datetime.now() - x) < datetime.timedelta(weeks = 1))]
        else:
            bad_days_df = pd.DataFrame({'day':[], 'date_registered':[]})

        bad_days_df.drop_duplicates(['day'], inplace=True)

        return bad_days_df

    def handle_flight_data_query(self, airports, dates_unix, bad_days_df):
        """
        Manage the query execution by connecting to the client,
        executing the query, and closing the connection.

        Parameters:
        - airports (dict): Dictionary containing departure and arrival airports.
        - dates_unix (dict): Dictionary containing start and end dates in UNIX timestamp format.
        - bad_days_df (pd.DataFrame): DataFrame containing days
        (UNIX timestamp) to exclude from query.

        Returns:
        - dict: Dictionary containing standard output and
        standard error from the query execution.
        """

        self.client.connect(
            self.credentials['hostname'],
            port=self.credentials['port'],
            username=self.credentials['username'],
            password=self.credentials['password']
        )

        query = self.create_query_command_for_flight_data(
            airports, dates_unix, bad_days_df['day'].to_list())

        self.log_verbose(f"Querying: {query}")
        query_results = self.execute_query(query)

        self.client.close()

        # Continue querying until successful or all bad days are excluded
        while "Disk I/O error" in query_results['stderr']:
            self.log_verbose("Bad day found, trying again.")
            bad_days_new = pd.DataFrame(
                {'day':[int(query_results['stderr'].split("\n")[3].split(
                "day=")[-1].split("/")[0])],
                 'date_registered':[datetime.datetime.now()]})

            if len(bad_days_df) == 0:
                bad_days_df = bad_days_new.copy()
            else:
                bad_days_df = pd.concat([bad_days_df, bad_days_new])
            bad_days_df.sort_values('day', inplace=True)
            bad_days_df.drop_duplicates(['day'], inplace=True)
            self.log_verbose("Bad Days:")
            for day in bad_days_df['day'].to_list():
                self.log_verbose(" - " + datetime.datetime.fromtimestamp(int(day)).strftime(
                    "%Y-%m-%d HH:MM:SS"))

            if self.bad_days_csv:
                bad_days_df.to_csv(self.bad_days_csv)

            self.client.connect(
                self.credentials['hostname'],
                port=self.credentials['port'],
                username=self.credentials['username'],
                password=self.credentials['password']
            )

            query = self.create_query_command_for_flight_data(
                airports, dates_unix, bad_days_df['day'].to_list())

            self.log_verbose(f"Querying: {query}")
            query_results = self.execute_query(query)

            self.client.close()
        return query_results, bad_days_df

    def query_flight_data(self, airports, dates):
        """
        Query flight data from the OpenSky database using the provided client.

        Parameters:
        - departure_airport (str): ICAO code of the departure airport.
        - arrival_airport (str): ICAO code of the arrival airport.
        - start_date: Start date, can be a date string, datetime.datetime object,
        UNIX integer or string, or datetime.date object.
        - end_date: Start date, can be a date string, datetime.datetime object,
        UNIX integer or string, or datetime.date object.

        Returns:
        - pd.DataFrame: DataFrame containing the flight data results.
        """
        # Set the signal handler and a 300-second alarm
        signal.signal(signal.SIGALRM, handler)
        # If logger is NOT None, checks if it's an isnstance of utils.Logger
        if self.logger and not isinstance(self.logger, utils.Logger):
            raise ValueError("Expected logger to be an instance of utils.Logger")

        bad_days_df = self.initialize_bad_days_df()

        # Convert dates to UNIX timestamps
        dates_unix = {'start': utils.to_unix_timestamp(dates['start']),
                     'end':utils.to_unix_timestamp(dates['end'])}

        dates_str = {'start':datetime.datetime.fromtimestamp(dates_unix['start']).strftime(
            "%Y-%m-%d HH:MM:SS"),
                    'end':datetime.datetime.fromtimestamp(dates_unix['end']).strftime(
            "%Y-%m-%d HH:MM:SS")}

        dates = self.compute_date_intervals(dates_unix)

        results_df = None
        for date_start, date_end in zip(dates[:-1], dates[1:]):
            dates_unix = {'start': date_start,
                     'end':date_end}

            dates_str = {'start':datetime.datetime.fromtimestamp(dates_unix['start']).strftime(
                "%Y-%m-%d HH:MM:SS"),
                        'end':datetime.datetime.fromtimestamp(dates_unix['end']).strftime(
                "%Y-%m-%d HH:MM:SS")}

            # Logs the initial query
            self.log_verbose(
                f"Querying data for flights from {airports['departure_airport']} \
to {airports['arrival_airport']} between the dates {dates_str['start']} and {dates_str['end']}"
)
            for _ in range(self.credentials['flight_data_retries']):
                try:
                    signal.alarm(self.credentials['flight_data_timeout'])
                    query_results, bad_days_df = self.handle_flight_data_query(
                        airports, dates_unix, bad_days_df)
                    break
                except TimeoutError:
                    self.log_verbose("Operation timed out, retrying...")
                finally:
                    signal.alarm(0)

            query_results, bad_days_df = self.handle_flight_data_query(
                        airports, dates_unix, bad_days_df)

            if results_df is not None:
                if len(results_df) == 0:
                    results_df = utils.parse_to_dataframe(query_results['stdout'])
                elif len(utils.parse_to_dataframe(query_results['stdout'])) != 0:
                    results_df = pd.concat([results_df, utils.parse_to_dataframe(
                        query_results['stdout'])])

            else:
                results_df = utils.parse_to_dataframe(query_results['stdout'])

        return results_df
    
    def handle_state_vector_query(self, icao24, start_time, end_time):
        """
        Execute a query to retrieve state vectors and handle query-related issues, 
        such as disk I/O errors caused by bad hours in the query timeframe, 
        by re-querying excluding identified problematic hours.

        The method attempts to retrieve state vectors for a specific aircraft
        (identified by its ICAO24 address) during a particular time range,
        while managing connection, query execution, and disconnection from the client.
        If a disk I/O error occurs (often signifying an issue with specific hours 
        in the database), the method identifies the problematic hour(s),
        logs them as "bad hours", and re-executes the query excluding these hours.

        Parameters:
        - icao24 (str): The ICAO24 address of the aircraft.
        - start_time (int): The UNIX timestamp representing the start of the time range.
        - end_time (int): The UNIX timestamp representing the end of the time range.

        Returns:
        - dict: A dictionary containing 'stdout' and 'stderr' from the query execution.
                'stdout' contains the query results, and 'stderr' contains error messages
                generated during the query, if any.
        """

        bad_hours = []
        # Connecting to client
        self.client.connect(
            self.credentials['hostname'],
            port=self.credentials['port'],
            username=self.credentials['username'],
            password=self.credentials['password'],
        )

        # Building the query
        query = self.create_query_command_for_state_vectors(
            icao24, {'start': start_time, 'end': end_time}, bad_hours)

        # Logs query
        self.log_verbose(f"Querying: {query}")

        # Execute the query
        query_results = {}
        _, query_results['stdout'], query_results['stderr'] \
        = self.client.exec_command(f"-q {query}")

        # Reads and decodes query results
        query_results['stdout'] = query_results['stdout'].read().decode()
        query_results['stderr'] = query_results['stderr'].read().decode()

        # Closing client
        self.client.close()

        # Continue querying until successful or all bad hours are excluded
        while "Disk I/O error" in query_results['stderr']:
            self.log_verbose("Bad hour found, trying again.")
            bad_hours += [int(query_results['stderr'].split("\n")[3].split(
                "hour=")[-1].split("/")[0])]
            bad_hours = sorted(bad_hours)
            self.log_verbose("Bad Hours:")
            for hour in bad_hours:
                date_str = datetime.datetime.fromtimestamp(hour).strftime("%Y-%m-%d")
                self.log_verbose(f" - {date_str}")
            # Re-query
            self.client.connect(
                self.credentials['hostname'],
                port=self.credentials['port'],
                username=self.credentials['username'],
                password=self.credentials['password']
            )
            query = self.create_query_command_for_state_vectors(
                icao24, {'start': start_time, 'end': end_time}, bad_hours)

            self.log_verbose(f"Querying: {query}")
            _, query_results['stdout'], query_results['stderr'] \
            = self.client.exec_command(f"-q {query}")

            query_results['stdout'] = query_results['stdout'].read().decode()
            query_results['stderr'] = query_results['stderr'].read().decode()

            self.client.close()
        return query_results

    def query_state_vectors(self, icao24, start_time, end_time):
        """
        Query state vectors data from the OpenSky database for a specific aircraft.

        Parameters:
        - icao24 (str): ICAO 24-bit address of the aircraft.
        - start_time: Start time, can be a date string, datetime.datetime object,
        UNIX integer or string, or datetime.date object.
        - end_time: End time, can be a date string, datetime.datetime object,
        UNIX integer or string, or datetime.date object.

        Returns:
        - pd.DataFrame: DataFrame containing the state vectors results.
        """

        # If logger is NOT None, checks if it's an isnstance of utils.Logger
        if self.logger and not isinstance(self.logger, utils.Logger):
            raise ValueError("Expected logger to be an instance of utils.Logger")

        # Convert dates to UNIX timestamps
        times_unix = {'start': utils.to_unix_timestamp(start_time),
                     'end': utils.to_unix_timestamp(end_time)}
        times_str = {'start': datetime.datetime.fromtimestamp(times_unix['start']).strftime(
            "%Y-%m-%d"),
                    'end': datetime.datetime.fromtimestamp(times_unix['end']).strftime(
            "%Y-%m-%d")}

        # Logs the initial query
        self.log_verbose(
            f"Querying data for statevectors for ICAO24 {icao24} \
between the times {times_str['start']} and {times_str['end']}"
        )

        for _ in range(self.credentials['state_vector_retries']):
            try:
                signal.alarm(self.credentials['flight_data_timeout'])
                query_results = self.handle_state_vector_query(
                    icao24, times_unix['start'], times_unix['end'])
                break
            except TimeoutError:
                self.log_verbose("Operation timed out, retrying...")
            finally:
                signal.alarm(0)

        return utils.parse_to_dataframe(query_results['stdout'])
