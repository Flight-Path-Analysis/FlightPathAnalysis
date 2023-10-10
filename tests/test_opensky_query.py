import sys
sys.path.append("../")
from src.backend import opensky_query
from unittest.mock import patch
from tests.mocks import mock_paramiko
import pandas as pd
import pytest

def test_query_flight_data():
    with patch('src.backend.opensky_query.paramiko', new=mock_paramiko):
        # ----------------Defining Good and Bad Cases---------------- #
        # Good Case 1, getting dataframe back
        
        good_credentials = {'username':'admin', 'password': 'password', 'hostname': 'ssh.mock.fake', 'port': '0'}

        expected_df = pd.DataFrame({'firstseen': {0: 1685580348, 1: 1685635052, 2: 1685651296, 3: 1685658580, 4: 1685667930, 5: 1685704862, 6: 1685720872, 7: 1685733665, 8: 1685745530, 9: 1685753450, 10: 1685790696, 11: 1685806879, 12: 1685821356}, 'lastseen': {0: 1685583833, 1: 1685638701, 2: 1685654994, 3: 1685662333, 4: 1685671930, 5: 1685708876, 6: 1685725434, 7: 1685737318, 8: 1685749472, 9: 1685757323, 10: 1685794703, 11: 1685810528, 12: 1685825333}, 'callsign': {0: 'ENY3479', 1: 'ENY3575', 2: 'ENY3431', 3: 'SKW4906', 4: 'SKW3021', 5: 'ENY3664', 6: 'ENY3704', 7: 'ENY3431', 8: 'SKW4906', 9: 'SKW3021', 10: 'ENY3664', 11: 'ENY3704', 12: 'JIA5074'}, 'icao24': {0: 'a1cd4e', 1: 'a1c229', 2: 'a2d6a2', 3: 'aa5d23', 4: 'a99686', 5: 'a24782', 6: 'a24782', 7: 'a2ddac', 8: 'aa0a6e', 9: 'aa11dc', 10: 'a24782', 11: 'a214de', 12: 'a7c878'}, 'estdepartureairport': {0: 'KBTR', 1: 'KBTR', 2: 'KBTR', 3: 'KBTR', 4: 'KBTR', 5: 'KBTR', 6: 'KBTR', 7: 'KBTR', 8: 'KBTR', 9: 'KBTR', 10: 'KBTR', 11: 'KBTR', 12: 'KBTR'}, 'estarrivalairport': {0: 'KDFW', 1: 'KDFW', 2: 'KDFW', 3: 'KDFW', 4: 'KDFW', 5: 'KDFW', 6: 'KDFW', 7: 'KDFW', 8: 'KDFW', 9: 'KDFW', 10: 'KDFW', 11: 'KDFW', 12: 'KDFW'}, 'day': {0: 1685577600, 1: 1685577600, 2: 1685577600, 3: 1685577600, 4: 1685664000, 5: 1685664000, 6: 1685664000, 7: 1685664000, 8: 1685664000, 9: 1685750400, 10: 1685750400, 11: 1685750400, 12: 1685750400}})
        
        # Bad Case 1, wrong username
        
        bad_credentials_1 = {'username':'user', 'password': 'password', 'hostname': 'ssh.mock.fake', 'port': '0'}
        
        output_bad_1 = pytest.raises(ValueError, match='Invalid Username')
        
        # Bad Case 2, wrong password
        
        bad_credentials_2 = {'username':'admin', 'password': 'pswrd', 'hostname': 'ssh.mock.fake', 'port': '0'}
        
        output_bad_2 = pytest.raises(ValueError, match='Invalid Password')
        
        # Bad Case 3, wrong port
        
        bad_credentials_3 = {'username':'admin', 'password': 'password', 'hostname': 'ssh.mock.fake', 'port': '1'}
        
        output_bad_3 = pytest.raises(ValueError, match='Invalid Port')
        
        # Bad Case 4, wrong port
        
        bad_credentials_4 = {'username':'admin', 'password': 'password', 'hostname': 'mock.fake', 'port': '0'}
        
        output_bad_4 = pytest.raises(ValueError, match='Invalid Hostname')
        
        
        # ----------------Testing Good and Bad Cases---------------- #
        
        # Good Case
        
        acquired_df = opensky_query.Querier(
            good_credentials['username'], 
            good_credentials['password'],
            good_credentials['hostname'],
            good_credentials['port']).query_flight_data('KBTR', 'KDFW', 1685577600, 1685836800)
        
        pd.testing.assert_frame_equal(acquired_df, expected_df)
        
        # Bad Case 1, wrong username
        
        with output_bad_1:
            opensky_query.Querier(
            bad_credentials_1['username'], 
            bad_credentials_1['password'],
            bad_credentials_1['hostname'],
            bad_credentials_1['port']).query_flight_data('KBTR', 'KDFW', 1685577600, 1685836800)
        
        # Bad Case 2, wrong password

        with output_bad_2:
            opensky_query.Querier(
            bad_credentials_2['username'], 
            bad_credentials_2['password'],
            bad_credentials_2['hostname'],
            bad_credentials_2['port']).query_flight_data('KBTR', 'KDFW', 1685577600, 1685836800)

        # Bad Case 3, wrong port

        with output_bad_3:
            opensky_query.Querier(
            bad_credentials_3['username'], 
            bad_credentials_3['password'],
            bad_credentials_3['hostname'],
            bad_credentials_3['port']).query_flight_data('KBTR', 'KDFW', 1685577600, 1685836800)

        # Bad Case 4, wrong hostname

        with output_bad_4:
            opensky_query.Querier(
            bad_credentials_4['username'], 
            bad_credentials_4['password'],
            bad_credentials_4['hostname'],
            bad_credentials_4['port']).query_flight_data('KBTR', 'KDFW', 1685577600, 1685836800)
        
def test_create_query_command_for_flight_data():
    # create_query_command_for_flight_data(self, departure_airport, arrival_airport, start_date_unix, end_date_unix, bad_days, limit=None)
    with patch('src.backend.opensky_query.paramiko', new=mock_paramiko):
        
        good_credentials = {'username':'admin', 'password': 'password', 'hostname': 'ssh.mock.fake', 'port': '0'}
        
        # ----------------Defining Good and Bad Cases---------------- #
        
        # Good Case 1, correct in and out
        
        good_input_1 = {'departure_airport': 'KBTR', 
                        'arrival_airport': 'KDFW', 
                        'start_date_unix': 1685577600, 
                        'end_date_unix': 1685836800,
                        'bad_days': []}
        
        good_output_1 = """\
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    ORDER BY firstseen;"""
        
        # Good Case 2, correct in and out with a bad day
        
        good_input_2 = {'departure_airport': 'KBTR', 
                        'arrival_airport': 'KDFW', 
                        'start_date_unix': 1685577600, 
                        'end_date_unix': 1685836800,
                        'bad_days': [1685577610]}
        
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
        
        good_input_3 = {'departure_airport': 'KBTR', 
                        'arrival_airport': 'KDFW', 
                        'start_date_unix': 1685577600, 
                        'end_date_unix': 1685836800,
                        'bad_days': [1685577610, 1685577630, 1685577410]}
        
        good_output_3 = """\
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    AND day != 1685577410
AND day != 1685577610
AND day != 1685577630
ORDER BY firstseen;"""
        
        
        
        # ----------------Testing Good and Bad Cases---------------- #
        
        # Good Case 1
        
        output = opensky_query.Querier(
            good_credentials['username'], 
            good_credentials['password'],
            good_credentials['hostname'],
            good_credentials['port']).create_query_command_for_flight_data(
                good_input_1['departure_airport'], 
                good_input_1['arrival_airport'], 
                good_input_1['start_date_unix'],
                good_input_1['end_date_unix'],
                bad_days = good_input_1['bad_days'])
        
        assert output == good_output_1
        
        # Good Case 2
        
        output = opensky_query.Querier(
            good_credentials['username'], 
            good_credentials['password'],
            good_credentials['hostname'],
            good_credentials['port']).create_query_command_for_flight_data(
                good_input_2['departure_airport'], 
                good_input_2['arrival_airport'], 
                good_input_2['start_date_unix'],
                good_input_2['end_date_unix'],
                bad_days = good_input_2['bad_days'])
        
        assert output == good_output_2
        
        # Good Case 3
        
        output = opensky_query.Querier(
            good_credentials['username'], 
            good_credentials['password'],
            good_credentials['hostname'],
            good_credentials['port']).create_query_command_for_flight_data(
                good_input_3['departure_airport'], 
                good_input_3['arrival_airport'], 
                good_input_3['start_date_unix'],
                good_input_3['end_date_unix'],
                bad_days = good_input_3['bad_days'])
        
        assert output == good_output_3
        
        