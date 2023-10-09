from src.backend.utils import to_number, parse_to_dataframe, to_unix_timestamp
import pandas as pd
import pytest
import re
import datetime

def test_to_number():
    # Test conversion of string to integer
    assert to_number("45") == 45
    
    # Test conversion of string to float
    assert to_number("3.14") == 3.14
    
    # Test string that can't be converted to number
    assert to_number("hello") == "hello"
    
    # Test non-string inputs
    assert to_number(42) == 42  # Integer input
    assert to_number(4.2) == 4.2  # Float input
    
    # Test other data types that can't be converted
    assert to_number(["1", "2", "3"]) == ["1", "2", "3"]  # List input
    assert to_number({"key": "value"}) == {"key": "value"}  # Dictionary input

    # Test edge cases
    assert to_number("") == ""  # Empty string
    assert to_number("  ") == "  "  # Whitespace string
    assert to_number(None) == None  # None input

def test_parse_to_dataframe():
    # ----------------Defining Good and Bad Cases---------------- #
    # Sample Input for good_case_1, sample result
    
    input_good_1 = """\
+----------+--------+------------+------------+---------------------+-------------------+
 | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
 | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
+----------+--------+------------+------------+---------------------+-------------------+\
"""
    
    output_good_1 = pd.DataFrame({
        'callsign': ['SKW5466'],
        'icao24': ['aced8f'],
        'firstseen': [1676665124],
        'lastseen': [1676667470],
        'estdepartureairport': ['KBTR'],
        'estarrivalairport': ['NULL']
    })
    
    # Sample input for good_case_1, no data result
    input_good_2 = """\
+----------+--------+------------+------------+---------------------+-------------------+
 | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
+----------+--------+------------+------------+---------------------+-------------------+\
"""

    output_good_2 =pd.DataFrame({
        'callsign': [],
        'icao24': [],
        'firstseen': [],
        'lastseen': [],
        'estdepartureairport': [],
        'estarrivalairport': []
    })
    
    # Sample input for bad_case_1, non-string input
    
    input_bad_1 = 5
    
    output_bad_1 = pytest.raises(ValueError, match="Provided results is not in string format.")
    
    # Sample input for bad_case_2, empty string input
    
    input_bad_2 = ''
    
    output_bad_2 = pytest.raises(ValueError, match="Provided results string is empty. Returning an empty DataFrame.")
    
    # Sample input for bad_case_3, arbitrary string input
    input_bad_3 = 'a'
    
    output_bad_3 = pytest.raises(ValueError, match=f"Invalid input, results should consist of at least 4 lines, even if empty of data.{input_bad_3}")
    
    # Sample input for bad_case_4, truncated results
    
    input_bad_4 = """\
+----------+--------+------------+------------+---------------------+-------------------+
 | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
 | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
 | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
 | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |\
"""
    
    output_bad_4 = pytest.raises(ValueError, match=re.escape("Invalid input, first, third, and last line is expected to contain nothing but \"+\" and \"-\""))
    
    # Sample input for bad_case_5, bad data
    
    input_bad_5 = """\
+----------+--------+------------+------------+---------------------+-------------------+
 | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
 | SKW5466  | 1676665124 | 1676667470 | KBTR                | NULL              |
+----------+--------+------------+------------+---------------------+-------------------+\
"""
    
    output_bad_5 = pytest.raises(ValueError, match=re.escape(f"Invalid input, data line 0 does not agree with columns format\n | SKW5466  | 1676665124 | 1676667470 | KBTR                | NULL              |"))
    
    # Testing for bad_case_6, bad data row
    
    input_bad_6 = """\
+----------+--------+------------+------------+---------------------+-------------------+
 | callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
| SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
+----------+--------+------------+------------+---------------------+-------------------+\
"""
    
    output_bad_6 = pytest.raises(ValueError, match=re.escape(f"Data lines should start with \" |\" and end with \"|\". 0, | SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |"))
    
    
    # ----------------Testing Good and Bad Cases---------------- #
    
    # Testing for input_good_1, sample result
    
    parsed_df = parse_to_dataframe(input_good_1)
    
    pd.testing.assert_frame_equal(parsed_df, output_good_1)
    
    # Testing for input_good_2, no-data result
    
    parsed_df = parse_to_dataframe(input_good_2)
    
    pd.testing.assert_frame_equal(parsed_df, output_good_2)
    
    # Testing for input_bad_1, non-string input
    
    with output_bad_1:
        parse_to_dataframe(input_bad_1)
        
    # Testing for input_bad_2, empty string input
    
    with output_bad_2:
        parse_to_dataframe(input_bad_2)
        
    # Testing for input_bad_3, arbitrary string input
    
    with output_bad_3:
        parse_to_dataframe(input_bad_3)
        
    # Testing for input_bad_4, truncated results
    
    with output_bad_4:
        parse_to_dataframe(input_bad_4)
                                 
    # Testing for input_bad_5, bad data
    with output_bad_5:
        parse_to_dataframe(input_bad_5)
        
    # Testing for input_bad_6, bad data row
    with output_bad_6:
        parse_to_dataframe(input_bad_6)
        
    
def test_to_unix_timestamp():
    # ----------------Defining Good and Bad Cases---------------- #
    input_good_1 = "2022-01-01"
    
    input_good_2 = "2022-01-01 00:00:00"
    
    input_good_3 = "1641016800"
    
    input_good_4 = 1641016800
    
    input_good_5 = 1641016800.0
    
    input_good_5 = datetime.date(2022, 1, 1)
    
    input_good_6 = datetime.datetime(2022, 1, 1, 0, 0)
    
    output_good = 1641016800
    
    input_bad_1 = "a"
    
    input_bad_2 = None

    output_bad =  pytest.raises(ValueError, match=re.escape("Unsupported date format"))
    
    # ----------------Testing Good and Bad Cases---------------- #
    
    assert to_unix_timestamp(input_good_1) == output_good
    
    assert to_unix_timestamp(input_good_2) == output_good
    
    assert to_unix_timestamp(input_good_3) == output_good
    
    assert to_unix_timestamp(input_good_4) == output_good
    
    assert to_unix_timestamp(input_good_5) == output_good
    
    assert to_unix_timestamp(input_good_6) == output_good
    
    with output_bad:
        to_unix_timestamp(input_bad_1)
        
    with output_bad:
        to_unix_timestamp(input_bad_2)
    
