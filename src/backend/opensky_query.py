import paramiko
import pandas as pd
import utils
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

    Methods:
    - create_query_command_for_flight_data: Generates the SQL query for flight data.
    - create_query_command_for_state_vectors: Generates the SQL query for state vectors.
    - query_flight_data: Executes the query and returns the results as a DataFrame.
    - query_state_vectors: Executes the query and returns the results as a DataFrame.
    """
    
    def __init__(self, username, password, hostname, port):
        """
        Initialize an instance of the Querier class.

        Parameters:
        - username (str): SSH username.
        - password (str): SSH password.
        - hostname (str): SSH hostname.
        - port (int): SSH port number.
        """
        self.__username = username
        self.__password = password
        self.hostname = hostname
        self.port = port
        self.client = paramiko.SSHClient()
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
        query =  f"""SELECT callsign, icao24, firstseen, lastseen, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = '{departure_airport}' 
    AND estarrivalairport = '{arrival_airport}'
    AND day >= {start_date_unix}
    AND day <= {end_date_unix}
    """
        # Add conditions to exclude bad days
        for day in bad_days:
            query += f'AND day != {day}\n'

        # Add ordering and limit if specified
        query += 'ORDER BY firstseen\n'
        if limit is None:
            query += ';'
        else:
            query += f'LIMIT {limit};'
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
    AND estarrivalairport = '{arrival_airport}'
    AND (time >= {start_time_unix} AND time <= {end_time_unix})
    AND (hour > {start_time_unix - 3600} AND hour < {end_time_unix + 3600})
    """
        # Add conditions to exclude bad hours
        for hour in bad_hours:
            query += f'AND hour != {hour}\n'

        # Add ordering and limit if specified
        query += 'ORDER BY time\n'
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
        # Convert dates to UNIX timestamps
        start_date_unix = utils.to_unix_timestamp(start_date)
        end_date_unix = utils.to_unix_timestamp(end_date)

        # Expected error message on successful query execution
        error_message_on_success = 'Starting Impala Shell without Kerberos authentication\n'
        errors = ''
        bad_days = []

        # Continue querying until successful or all bad days are excluded
        while errors != error_message_on_success:
            self.client.connect(self.hostname, port=self.port, username=self.__username, password=self.__password)
            
            query = self.create_query_command_for_flight_data(departure_airport, arrival_airport, start_date_unix, end_date_unix, bad_days)
            print("Querying: ", query)

            # Execute the query
            cmd = "-q " + query
            stdin, stdout, stderr = self.client.exec_command(cmd)
            results = stdout.read().decode()
            errors = stderr.read().decode()

            # If Disk I/O error is found, add the day to bad_days list
            if 'Disk I/O error' in errors:
                print("Bad day found, trying again.")
                bad_days += [eval(errors.split('\n')[3].split('day=')[-1].split('/')[0])]
                bad_days = sorted(bad_days)
                print("Bad Days:")
                for day in bad_days:
                    print(' - ', datetime.datetime.fromtimestamp(day).strftime('%Y-%m-%d'))

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

        Returns:
        - pd.DataFrame: DataFrame containing the state vectors results.
        """
        self.client.connect(self.hostname, port=self.port, username=self.__username, password=self.__password)
            
            query = self.create_query_command_for_state_vectors(icao24, start_time_unix, end_time_unix, bad_hours)
            print("Querying: ", query)

            # Execute the query
            cmd = "-q " + query
            stdin, stdout, stderr = self.client.exec_command(cmd)
            results = stdout.read().decode()
            errors = stderr.read().decode()

            # If Disk I/O error is found, add the hour to bad_hours list
            if 'Disk I/O error' in errors:
                print("Bad hour found, trying again.")
                bad_hours += [eval(errors.split('\n')[3].split('hour=')[-1].split('/')[0])]
                bad_hours = sorted(bad_hours)
                print("Bad Hours:")
                for hour in bad_hours:
                    print(' - ', datetime.datetime.fromtimestamp(hour).strftime('%Y-%m-%d HH:MM:SS'))

            self.client.close()
        # Convert the result to a DataFrame and return
        return utils.parse_to_dataframe(results)
        