"""
Unit tests for the opensky_query module from src.backend.

This module contains test functions for validating the behavior of the methods:
- `query_flight_data` from the `Querier` class in `opensky_query`, and
- `create_query_command_for_flight_data` from the same class.

The tests use a mock for the paramiko module, which is an interface to work with
SSH servers. The purpose of these tests is to ensure that the flight data queries
are constructed correctly and that SSH communication behaves as expected given different inputs.
"""

from unittest.mock import patch
import pandas as pd
import pytest
from tests.mocks import mock_paramiko
from src.backend import opensky_query
from src.backend import utils
import datetime
import os

def test_query_flight_data():
    """
    Test the `query_flight_data` method from the `Querier` class in `opensky_query`.
    
    This test function covers:
    - Successful retrieval of flight data given valid credentials and query parameters.
    - Handling of invalid credentials like wrong username, password, port, and hostname.
    - Checks the structure and content of the returned DataFrame against expected outputs.
    """
    with patch("src.backend.opensky_query.paramiko", new=mock_paramiko):
        # ----------------Defining Good and Bad Cases---------------- #
        # Good Case 1, getting dataframe back

        good_credentials = {
            "username": "admin",
            "password": "password",
            "hostname": "ssh.mock.fake",
            "port": "0",
            "chunk_size": 100000000,
            "flight_data_retries": 3,
            "flight_data_timeout": 300,
            "state_vector_retries": 3,
            "state_vector_timeout": 300
        }

        expected_df = pd.DataFrame(
            {
                "firstseen": {
                    0: 1685580348,
                    1: 1685635052,
                    2: 1685651296,
                    3: 1685658580,
                    4: 1685667930,
                    5: 1685704862,
                    6: 1685720872,
                    7: 1685733665,
                    8: 1685745530,
                    9: 1685753450,
                    10: 1685790696,
                    11: 1685806879,
                    12: 1685821356,
                },
                "lastseen": {
                    0: 1685583833,
                    1: 1685638701,
                    2: 1685654994,
                    3: 1685662333,
                    4: 1685671930,
                    5: 1685708876,
                    6: 1685725434,
                    7: 1685737318,
                    8: 1685749472,
                    9: 1685757323,
                    10: 1685794703,
                    11: 1685810528,
                    12: 1685825333,
                },
                "callsign": {
                    0: "ENY3479",
                    1: "ENY3575",
                    2: "ENY3431",
                    3: "SKW4906",
                    4: "SKW3021",
                    5: "ENY3664",
                    6: "ENY3704",
                    7: "ENY3431",
                    8: "SKW4906",
                    9: "SKW3021",
                    10: "ENY3664",
                    11: "ENY3704",
                    12: "JIA5074",
                },
                "icao24": {
                    0: "a1cd4e",
                    1: "a1c229",
                    2: "a2d6a2",
                    3: "aa5d23",
                    4: "a99686",
                    5: "a24782",
                    6: "a24782",
                    7: "a2ddac",
                    8: "aa0a6e",
                    9: "aa11dc",
                    10: "a24782",
                    11: "a214de",
                    12: "a7c878",
                },
                "estdepartureairport": {
                    0: "KBTR",
                    1: "KBTR",
                    2: "KBTR",
                    3: "KBTR",
                    4: "KBTR",
                    5: "KBTR",
                    6: "KBTR",
                    7: "KBTR",
                    8: "KBTR",
                    9: "KBTR",
                    10: "KBTR",
                    11: "KBTR",
                    12: "KBTR",
                },
                "estarrivalairport": {
                    0: "KDFW",
                    1: "KDFW",
                    2: "KDFW",
                    3: "KDFW",
                    4: "KDFW",
                    5: "KDFW",
                    6: "KDFW",
                    7: "KDFW",
                    8: "KDFW",
                    9: "KDFW",
                    10: "KDFW",
                    11: "KDFW",
                    12: "KDFW",
                },
                "day": {
                    0: 1685577600,
                    1: 1685577600,
                    2: 1685577600,
                    3: 1685577600,
                    4: 1685664000,
                    5: 1685664000,
                    6: 1685664000,
                    7: 1685664000,
                    8: 1685664000,
                    9: 1685750400,
                    10: 1685750400,
                    11: 1685750400,
                    12: 1685750400,
                },
            }
        )

        # Bad Case 1, wrong username

        bad_credentials_1 = {
            "username": "user",
            "password": "password",
            "hostname": "ssh.mock.fake",
            "port": "0",
            "chunk_size": 100000000,
            "flight_data_retries": 3,
            "flight_data_timeout": 300,
            "state_vector_retries": 3,
            "state_vector_timeout": 300
        }

        output_bad_1 = pytest.raises(ValueError, match="Invalid Username")

        # Bad Case 2, wrong password

        bad_credentials_2 = {
            "username": "admin",
            "password": "pswrd",
            "hostname": "ssh.mock.fake",
            "port": "0",
            "chunk_size": 100000000,
            "flight_data_retries": 3,
            "flight_data_timeout": 300,
            "state_vector_retries": 3,
            "state_vector_timeout": 300
        }

        output_bad_2 = pytest.raises(ValueError, match="Invalid Password")

        # Bad Case 3, wrong port

        bad_credentials_3 = {
            "username": "admin",
            "password": "password",
            "hostname": "ssh.mock.fake",
            "port": "1",
            "chunk_size": 100000000,
            "flight_data_retries": 3,
            "flight_data_timeout": 300,
            "state_vector_retries": 3,
            "state_vector_timeout": 300
        }

        output_bad_3 = pytest.raises(ValueError, match="Invalid Port")

        # Bad Case 4, wrong port

        bad_credentials_4 = {
            "username": "admin",
            "password": "password",
            "hostname": "mock.fake",
            "port": "0",
            "chunk_size": 100000000,
            "flight_data_retries": 3,
            "flight_data_timeout": 300,
            "state_vector_retries": 3,
            "state_vector_timeout": 300
        }

        output_bad_4 = pytest.raises(ValueError, match="Invalid Hostname")

        # ----------------Testing Good and Bad Cases---------------- #

        # Good Case

        acquired_df = opensky_query.Querier(good_credentials).query_flight_data(
            {'departure_airport': "KBTR", "arrival_airport": "KDFW"},
            {'start': 1685577600, 'end': 1685836800}
        )
        print('----\nAquired DF:' + str(acquired_df) + '\n----')
        print('----\nAquired DF:' + str(expected_df) + '\n----')

        pd.testing.assert_frame_equal(acquired_df, expected_df)

        # Bad Case 1, wrong username

        with output_bad_1:
            opensky_query.Querier(bad_credentials_1).query_flight_data(
            {'departure_airport':"KBTR", "arrival_airport": "KDFW"},
            {'start': 1685577600, 'end': 1685836800}
            )

        # Bad Case 2, wrong password

        with output_bad_2:
            opensky_query.Querier(bad_credentials_2).query_flight_data(
            {'departure_airport':"KBTR", "arrival_airport": "KDFW"},
            {'start': 1685577600, 'end': 1685836800}
            )

        # Bad Case 3, wrong port

        with output_bad_3:
            opensky_query.Querier(bad_credentials_3).query_flight_data(
            {'departure_airport':"KBTR", "arrival_airport": "KDFW"},
            {'start': 1685577600, 'end': 1685836800}
            )

        # Bad Case 4, wrong hostname

        with output_bad_4:
            opensky_query.Querier(bad_credentials_4).query_flight_data(
            {'departure_airport':"KBTR", "arrival_airport": "KDFW"},
            {'start': 1685577600, 'end': 1685836800}
            )
test_query_flight_data()
def test_create_query_command_for_flight_data():
    """
    Test the `create_query_command_for_flight_data` method from the `Querier`
    class in `opensky_query`.
    
    This test function ensures that:
    - The query commands are constructed correctly based on the given parameters.
    - The commands exclude specific 'bad days' when required.
    - The overall structure and syntax of the returned SQL command matches the expected format.
    """
    with patch("src.backend.opensky_query.paramiko", new=mock_paramiko):
        good_credentials = {
            "username": "admin",
            "password": "password",
            "hostname": "ssh.mock.fake",
            "port": "0",
            "chunk_size": 100000000
        }

        # ----------------Defining Good and Bad Cases---------------- #

        # Good Case 1, correct in and out

        good_input_1 = {
            "airports": {"departure_airport": "KBTR", "arrival_airport": "KDFW"},
            "dates": {"start": 1685577600, "end": 1685836800},
            "bad_days": [],
        }

        good_output_1 = """\
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    ORDER BY firstseen;"""

        # Good Case 2, correct in and out with a bad day

        good_input_2 = {
            "airports": {"departure_airport": "KBTR", "arrival_airport": "KDFW"},
            "dates": {"start": 1685577600, "end": 1685836800},
            "bad_days": [1685577610],
        }

        good_output_2 = """\
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    AND day != 1685577610
ORDER BY firstseen;"""

        # Good Case 3, correct in and out with many days
        good_input_3 = {
            "airports": {"departure_airport": "KBTR", "arrival_airport": "KDFW"},
            "dates": {"start": 1685577600, "end": 1685836800},
            "bad_days": [1685577410, 1685577610, 1685577630],
        }

        good_output_3 = """\
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    AND day != 1685577610
AND day != 1685577630
ORDER BY firstseen;"""

        # ----------------Testing Good and Bad Cases---------------- #

        # Good Case 1

        output = opensky_query.Querier(
            good_credentials
        ).create_query_command_for_flight_data(
            good_input_1["airports"],
            good_input_1["dates"],
            bad_days=good_input_1["bad_days"],
        )

        assert output == good_output_1

        # Good Case 2

        output = opensky_query.Querier(
            good_credentials
        ).create_query_command_for_flight_data(
            good_input_2["airports"],
            good_input_2["dates"],
            bad_days=good_input_2["bad_days"],
        )

        assert output == good_output_2

        # Good Case 3

        output = opensky_query.Querier(
            good_credentials
        ).create_query_command_for_flight_data(
            good_input_3["airports"],
            good_input_3["dates"],
            bad_days=good_input_3["bad_days"],
        )

        assert output == good_output_3

def test_handler():
    """
    Test the `handler` function from the `opensky_query` module.
    
    This test function covers:
    - Raising a TimeoutError when the signal handler is called.
    """
    import signal
    
    # Set the signal handler to the `handler` function
    signal.signal(signal.SIGALRM, opensky_query.handler)
    
    # Set the alarm to go off in 1 second
    signal.alarm(1)
    
    # Wait for the alarm to go off and catch the TimeoutError
    with pytest.raises(TimeoutError):
        while True:
            pass
    
def test_initialize_bad_days_df_with_existing_csv():
    """
    Test the `initialize_bad_days_df` method from the `Querier` class in `opensky_query`
    when `self.bad_days_csv` exists and points to a valid path.
    
    This test function covers:
    - Reading the `self.bad_days_csv` file into a DataFrame.
    - Retaining only the entries from the last week.
    - Dropping duplicates based on the 'day' column.
    """
    good_credentials = {
        "username": "admin",
        "password": "password",
        "hostname": "ssh.mock.fake",
        "port": "0",
        "chunk_size": 100000000,
        "flight_data_retries": 3,
        "flight_data_timeout": 300,
        "state_vector_retries": 3,
        "state_vector_timeout": 300
    }
    # Create a temporary CSV file with some data
    temp_csv = "temp_bad_days.csv"
    df = {'day':[], 'date_registered':[]}
    now = datetime.datetime.now()
    for i in range(10):
        time = now - datetime.timedelta(days=i)
        year = time.year
        month = time.month
        day = time.day
        hh = time.hour
        mm = time.minute
        ss = time.second
        df['day'] += [utils.to_unix_timestamp(time)]
        df['date_registered'] += [f"{year}-{month}-{day} {hh}:{mm}:{ss}"]
    df = pd.DataFrame(df)
    df.to_csv(temp_csv)
    
    # Initialize the bad days DataFrame
    querier = opensky_query.Querier(good_credentials)
    querier.bad_days_csv = temp_csv
    print(pd.read_csv(querier.bad_days_csv))
    bad_days_df = querier.initialize_bad_days_df()

    print(bad_days_df)
    
    # Check that the DataFrame has the expected shape and contents
    assert bad_days_df.shape == (7, 2)
    
    # Clean up the temporary CSV file
    os.remove(temp_csv)


def test_initialize_bad_days_df_with_nonexistent_csv():
    """
    Test the `initialize_bad_days_df` method from the `Querier` class in `opensky_query`
    when `self.bad_days_csv` doesn't point to a valid path.
    
    This test function covers:
    - Creating an empty DataFrame with columns 'day' and 'date_registered'.
    - Dropping duplicates based on the 'day' column.
    """
    good_credentials = {
        "username": "admin",
        "password": "password",
        "hostname": "ssh.mock.fake",
        "port": "0",
        "chunk_size": 100000000,
        "flight_data_retries": 3,
        "flight_data_timeout": 300,
        "state_vector_retries": 3,
        "state_vector_timeout": 300
    }
    # Initialize the bad days DataFrame
    querier = opensky_query.Querier(good_credentials)
    querier.bad_days_csv = "nonexistent.csv"
    bad_days_df = querier.initialize_bad_days_df()
    
    # Check that the DataFrame has the expected shape and contents
    assert bad_days_df.shape == (0, 2)
    assert list(bad_days_df.columns) == ["day", "date_registered"]