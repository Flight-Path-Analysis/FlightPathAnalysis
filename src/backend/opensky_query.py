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
import datetime
import paramiko
import pandas as pd

from src.backend import utils

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

        self.__username = credentials['username']
        self.__password = credentials['password']
        self.hostname = credentials['hostname']
        self.port = credentials['port']
        self.client = paramiko.SSHClient()
        self.chunk_size = credentials['chunk_size']

        if 'bad_days_csv' in credentials:
            self.bad_days_csv = credentials['bad_days_csv']
        else:
            self.bad_days_csv = None

        self.logger = logger

        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

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

    def query_flight_data(
        self, airports, dates):
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

        # Function that will log text if logger is different than None
        def log_verbose(message):
            if self.logger:
                self.logger.log(message)

        # If logger is NOT None, checks if it's an isnstance of utils.Logger
        if self.logger and not isinstance(self.logger, utils.Logger):
            raise ValueError("Expected logger to be an instance of utils.Logger")
        
        # Array to save the days that return an error from the query
        if self.bad_days_csv and os.path.exists(self.bad_days_csv):
            bad_days_df = pd.read_csv(self.bad_days_csv, index_col = 0,
                                      parse_dates=['date_registered'])

            bad_days_df = bad_days_df[bad_days_df['date_registered'].apply(
                lambda x: (datetime.datetime.now() - x) < datetime.timedelta(weeks = 1))]
        else:
            bad_days_df = pd.DataFrame({'day':[], 'date_registered':[]})

        # Convert dates to UNIX timestamps
        dates_unix = {'start': utils.to_unix_timestamp(dates['start']),
                     'end':utils.to_unix_timestamp(dates['end'])}

        dates_str = {'start':datetime.datetime.fromtimestamp(dates_unix['start']).strftime(
            "%Y-%m-%d HH:MM:SS"),
                    'end':datetime.datetime.fromtimestamp(dates_unix['end']).strftime(
            "%Y-%m-%d HH:MM:SS")}
        
        if dates_unix['end'] - dates_unix['start'] <= self.chunk_size:
            dates = [dates_unix['start'], dates_unix['end']]
        else:
            dates = list(range(dates_unix['start'], dates_unix['end'], self.chunk_size))
            if dates[-1] != dates_unix['end']:
                dates += [dates_unix['end']]
                
        results_df = None
        for date_start, date_end in zip(dates[:-1], dates[1:]):
            dates_unix = {'start': date_start,
                     'end':date_end}

            dates_str = {'start':datetime.datetime.fromtimestamp(dates_unix['start']).strftime(
                "%Y-%m-%d HH:MM:SS"),
                        'end':datetime.datetime.fromtimestamp(dates_unix['end']).strftime(
                "%Y-%m-%d HH:MM:SS")}
            
            # Logs the initial query
            log_verbose(
                f"Querying data for flights from {airports['departure_airport']} \
to {airports['arrival_airport']} between the dates {dates_str['start']} and {dates_str['end']}"
            )

            # Connecting to client
            self.client.connect(
                self.hostname,
                port=self.port,
                username=self.__username,
                password=self.__password,
            )

            # Building the query
            query = self.create_query_command_for_flight_data(
                airports, dates_unix, bad_days_df['day'].to_list())

            # Logs query
            log_verbose(f"Querying: {query}")

            # Execute the query and save it to query_results
            query_results = {}
            _, query_results['stdout'], query_results['stderr'] \
            = self.client.exec_command(f"-q {query}")

            # Reads and decodes query results
            query_results['stdout'] = query_results['stdout'].read().decode()
            query_results['stderr'] = query_results['stderr'].read().decode()

            self.client.close()

            # Continue querying until successful or all bad days are excluded
            while "Disk I/O error" in query_results['stderr']:
                log_verbose("Bad day found, trying again.")
                
                bad_days_new = pd.DataFrame(
                    {'day':[int(query_results['stderr'].split("\n")[3].split(
                    "day=")[-1].split("/")[0])],
                     'date_registered':[datetime.datetime.now()]})
                
                if len(bad_days_df) == 0:
                    bad_days_df = bad_days_new.copy()
                else:
                    bad_days_df = pd.concat([bad_days_df, bad_days_new])
                    
                bad_days_df.sort_values('day', inplace=True)

                log_verbose("Bad Days:")
                for day in bad_days_df['day'].to_list():
                    log_verbose(" - " + datetime.datetime.fromtimestamp(int(day)).strftime(
                        "%Y-%m-%d HH:MM:SS"))

                if self.bad_days_csv:
                    bad_days_df.to_csv(self.bad_days_csv)

                self.client.connect(
                    self.hostname,
                    port=self.port,
                    username=self.__username,
                    password=self.__password)

                query = self.create_query_command_for_flight_data(
                    airports, dates_unix, bad_days_df['day'].to_list())
                log_verbose(f"Querying: {query}")
                _, query_results['stdout'], query_results['stderr'] \
                = self.client.exec_command(f"-q {query}")
                query_results['stdout'] = query_results['stdout'].read().decode()
                query_results['stderr'] = query_results['stderr'].read().decode()

                self.client.close()
                
            if results_df is not None:
                if len(results_df) == 0:
                    results_df = utils.parse_to_dataframe(query_results['stdout'])
                elif len(utils.parse_to_dataframe(query_results['stdout'])) != 0:
                    results_df = pd.concat([results_df, utils.parse_to_dataframe(query_results['stdout'])])
                    
            else:
                results_df = utils.parse_to_dataframe(query_results['stdout'])

        return results_df

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

        # Function that will log text if logger is different than None
        def log_verbose(message):
            if self.logger:
                self.logger.log(message)

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

        # Array to save the days that return an error from the query
        bad_hours = []

        # Logs the initial query
        log_verbose(
            f"Querying data for statevectors for ICAO24 {icao24} \
            between the times {times_str['start']} and {times_str['end']}"
        )

        # Connecting to client
        self.client.connect(
            self.hostname,
            port=self.port,
            username=self.__username,
            password=self.__password,
        )

        # Building the query
        query = self.create_query_command_for_state_vectors(
            icao24, times_unix, bad_hours)

        # Logs query
        log_verbose(f"Querying: {query}")

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
            log_verbose("Bad hour found, trying again.")
            bad_hours += [int(query_results['stderr'].split("\n")[3].split(
                "hour=")[-1].split("/")[0])]
            bad_hours = sorted(bad_hours)
            log_verbose("Bad Hours:")
            for hour in bad_hours:
                date_str = datetime.datetime.fromtimestamp(hour).strftime("%Y-%m-%d")
                log_verbose(f" - {date_str}")
            # Re-query
            self.client.connect(
                self.hostname,
                port=self.port,
                username=self.__username,
                password=self.__password,
            )
            query = self.create_query_command_for_state_vectors(
                icao24, times_unix, bad_hours)

            log_verbose(f"Querying: {query}")
            _, query_results['stdout'], query_results['stderr'] \
            = self.client.exec_command(f"-q {query}")

            query_results['stdout'] = query_results['stdout'].read().decode()
            query_results['stderr'] = query_results['stderr'].read().decode()

            self.client.close()
        return utils.parse_to_dataframe(query_results['stdout'])
