import paramiko
import pandas as pd
from src.backend import utils
import datetime

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
    
    def __init__(self, username, password, hostname, port, logger = None):
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
        
        self.__username = username
        self.__password = password
        self.hostname = hostname
        self.port = port
        self.client = paramiko.SSHClient()
        self.logger = logger
        
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def create_query_command_for_flight_data(self, departure_airport, arrival_airport, start_date_unix, end_date_unix, bad_days, limit=None):
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
        # Begin SQL command
        query =  f"""SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = '{departure_airport}' 
    AND estarrivalairport = '{arrival_airport}'
    AND day >= {start_date_unix}
    AND day <= {end_date_unix}
    """
        # Add conditions to exclude bad days
        for day in sorted(bad_days):
            query += f'AND day != {day}\n'

        # Add ordering and limit if specified
        query += 'ORDER BY firstseen'
        if limit is None:
            query += ';'
        else:
            query += f'\nLIMIT {limit};'
        return query

    def create_query_command_for_state_vectors(self, icao24, start_time_unix, end_time_unix, bad_hours, limit=None):
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

        query =  f"""SELECT time, lat, lon, velocity, heading, baroaltitude, geoaltitude, onground, hour
    FROM state_vectors_data4
    WHERE icao24 = '{icao24}' 
    AND (time >= {start_time_unix} AND time <= {end_time_unix})
    AND (hour > {start_time_unix - 3600} AND hour < {end_time_unix + 3600})
    """
        # Add conditions to exclude bad hours
        for hour in bad_hours:
            query += f'AND hour != {hour}\n'

        # Add ordering and limit if specified
        query += 'ORDER BY time;'
        return query
            
    def query_flight_data(self, departure_airport, arrival_airport, start_date, end_date):
        """
        Query flight data from the OpenSky database using the provided client.

        Parameters:
        - departure_airport (str): ICAO code of the departure airport.
        - arrival_airport (str): ICAO code of the arrival airport.
        - start_date: Start date, can be a date string, datetime.datetime object, UNIX integer or string, or datetime.date object.
        - end_date: Start date, can be a date string, datetime.datetime object, UNIX integer or string, or datetime.date object.

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

        # Convert dates to UNIX timestamps
        start_date_unix = utils.to_unix_timestamp(start_date)
        end_date_unix = utils.to_unix_timestamp(end_date)
        
        start_date_str = datetime.datetime.fromtimestamp(start_date_unix).strftime('%Y-%m-%d HH:MM:SS')
        end_date_str = datetime.datetime.fromtimestamp(end_date_unix).strftime('%Y-%m-%d HH:MM:SS')
        
        # Array to save the days that return an error from the query
        bad_days = []
        
        # Logs the initial query 
        log_verbose(f'Querying data for flights from {departure_airport} to {arrival_airport} between the dates {start_date_str} and {end_date_str}')
        
        # Connecting to client
        self.client.connect(self.hostname, port=self.port, username=self.__username, password=self.__password)
            
        # Building the query
        query = self.create_query_command_for_flight_data(departure_airport, arrival_airport, start_date_unix, end_date_unix, bad_days)
           
        # Logs query
        log_verbose(f"Querying: {query}")
        
        # Execute the query
        cmd = f'-q {query}'
        stdin, stdout, stderr = self.client.exec_command(cmd)
        results = stdout.read().decode()
        errors = stderr.read().decode()
        
        # Closing the client
        self.client.close()
        
        # Continue querying until successful or all bad days are excluded
        while 'Disk I/O error' in errors:
            log_verbose("Bad day found, trying again.")
            bad_days += [eval(errors.split('\n')[3].split('day=')[-1].split('/')[0])]
            bad_days = sorted(bad_days)
            log_verbose("Bad Days:")
            for day in bad_days:
                date_str = datetime.datetime.fromtimestamp(day).strftime('%Y-%m-%d HH:MM:SS')
                log_verbose(f' - {date_str}')

            self.client.connect(self.hostname, port=self.port, username=self.__username, password=self.__password)
            query = self.create_query_command_for_flight_data(departure_airport, arrival_airport, start_date_unix, end_date_unix, bad_days)
            log_verbose(f"Querying: {query}")
            cmd = f'-q {query}'
            stdin, stdout, stderr = self.client.exec_command(cmd)
            results = stdout.read().decode()
            errors = stderr.read().decode()
            self.client.close()
            
        # Convert the result to a DataFrame and return
        return utils.parse_to_dataframe(results)
    
    def query_state_vectors(self, icao24, start_time, end_time):
        """
        Query state vectors data from the OpenSky database for a specific aircraft.

        Parameters:
        - icao24 (str): ICAO 24-bit address of the aircraft.
        - start_time: Start time, can be a date string, datetime.datetime object, UNIX integer or string, or datetime.date object.
        - end_time: End time, can be a date string, datetime.datetime object, UNIX integer or string, or datetime.date object.
        - logger (utils.Logger, optional): Logger instance for verbose logging. Defaults to None.

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
        start_time_unix = utils.to_unix_timestamp(start_time)
        end_time_unix = utils.to_unix_timestamp(end_time)
        
        start_time_str = datetime.datetime.fromtimestamp(start_time_unix).strftime('%Y-%m-%d')
        end_time_str = datetime.datetime.fromtimestamp(end_time_unix).strftime('%Y-%m-%d')
        
        # Array to save the days that return an error from the query
        bad_hours = []

        # Logs the initial query 
        log_verbose(f'Querying data for statevectors for ICAO24 {icao24} between the times {start_time_str} and {end_time_str}')
            
        # Connecting to client
        self.client.connect(self.hostname, port=self.port, username=self.__username, password=self.__password)
        
        # Building the query
        query = self.create_query_command_for_state_vectors(icao24, start_time_unix, end_time_unix, bad_hours)
        
        # Logs query
        log_verbose(f"Querying: {query}")
        
        # Execute the query
        cmd = f'-q {query}'
        stdin, stdout, stderr = self.client.exec_command(cmd)
        results = stdout.read().decode()
        errors = stderr.read().decode()
        
        # Closing client
        self.client.close()
        
        # Continue querying until successful or all bad days are excluded
        while 'Disk I/O error' in errors:
            log_verbose("Bad hour found, trying again.")
            bad_hours += [eval(errors.split('\n')[3].split('hour=')[-1].split('/')[0])]
            bad_hours = sorted(bad_hours)
            log_verbose("Bad Hours:")
            for hour in bad_hours:
                date_str = datetime.datetime.fromtimestamp(day).strftime('%Y-%m-%d')
                log_verbose(f' - {date_str}')
            # Re-query
            self.client.connect(self.hostname, port=self.port, username=self.__username, password=self.__password)
            query = self.create_query_command_for_state_vectors(icao24, start_time_unix, end_time_unix, bad_hours)
            log_verbose(f"Querying: {query}")
            cmd = "-q " + query
            stdin, stdout, stderr = self.client.exec_command(cmd)
            results = stdout.read().decode()
            errors = stderr.read().decode()
            self.client.close()
        
        return utils.parse_to_dataframe(results)
    
    
