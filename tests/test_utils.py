"""
This module contains test functions for various utility functions from 
the `src.backend.utils` module.

The module tests the following utility functions:
- `to_number`: This function attempts to convert a given input to a number (integer or float).
- `parse_to_dataframe`: This function parses the text results of a database 
query to a pandas DataFrame.
- `to_unix_timestamp`: This function converts various date input formats to a UNIX timestamp.

Each utility function has its own testing function within this module:
- `test_to_number`: Tests the `to_number` function with various inputs.
- `test_parse_to_dataframe`: Tests the `parse_to_dataframe` function with various inputs.
- `test_to_unix_timestamp`: Tests the `to_unix_timestamp` function with various inputs.

Dependencies:
- datetime: Used for generating timestamp and date objects for tests.
- re: Used for regular expressions in test assertions.
- pytest: Used for defining and executing tests.
- pandas: Used for comparing the output DataFrame with expected results.
- src.backend.utils: The module containing the utility functions being tested.
"""

import datetime
import re
import pytest
import pandas as pd

from src.backend.utils import to_number, parse_to_dataframe, to_unix_timestamp


def test_to_number():
    """
    Test the `to_number` function from `src.backend.utils`.

    The function is tested for:
    - Conversion of strings to integers and floats.
    - Strings that cannot be converted to numbers.
    - Non-string inputs like integers, floats, lists, and dictionaries.
    - Edge cases such as empty strings, whitespace strings, and `None` inputs.
    """
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
    assert to_number(None) is None  # None input


def test_parse_to_dataframe():
    """
    Test the `parse_to_dataframe` function from `src.backend.utils`.

    The function is tested for:
    - Parsing correctly formatted query result strings.
    - Parsing empty query result strings.
    - Handling non-string input.
    - Handling empty string input.
    - Handling arbitrary strings that do not represent query results.
    - Parsing truncated query results.
    - Handling query result strings with incorrect data formatting.
    """
    # ----------------Defining Good and Bad Cases---------------- #
    # Sample Input for good_case_1, sample result

    input_good_1 = """\
+----------+--------+------------+------------+---------------------+-------------------+
| callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
| SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
+----------+--------+------------+------------+---------------------+-------------------+\
"""

    output_good_1 = pd.DataFrame(
        {
            "callsign": ["SKW5466"],
            "icao24": ["aced8f"],
            "firstseen": [1676665124],
            "lastseen": [1676667470],
            "estdepartureairport": ["KBTR"],
            "estarrivalairport": ["NULL"],
        }
    )

    # Sample input for good_case_1, no data result
    input_good_2 = """\
+----------+--------+------------+------------+---------------------+-------------------+
| callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
+----------+--------+------------+------------+---------------------+-------------------+\
"""

    output_good_2 = pd.DataFrame(
        {
            "callsign": [],
            "icao24": [],
            "firstseen": [],
            "lastseen": [],
            "estdepartureairport": [],
            "estarrivalairport": [],
        }
    )

    # Sample input for bad_case_1, non-string input

    input_bad_1 = 5

    output_bad_1 = pytest.raises(
        ValueError, match="Provided results is not in string format."
    )

    # Sample input for bad_case_2, empty string input

    input_bad_2 = ""

    output_bad_2 = pytest.raises(ValueError, match="Provided results string is empty.")

    # Sample input for bad_case_3, arbitrary string input
    input_bad_3 = "a"

    output_bad_3 = pytest.raises(
        ValueError,
        match=f"Invalid input, results should consist of at \
least 4 lines, even if empty of data.{input_bad_3}",
    )

    # Sample input for bad_case_4, truncated results

    input_bad_4 = """\
+----------+--------+------------+------------+---------------------+-------------------+
| callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
| SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
| SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |
| SKW5466  | aced8f | 1676665124 | 1676667470 | KBTR                | NULL              |\
"""

    output_bad_4 = pytest.raises(
        ValueError,
        match=re.escape(
            'Invalid input, first, third, and last line is \
expected to contain nothing but "+" and "-"'
        ),
    )

    # Sample input for bad_case_5, bad data

    input_bad_5 = """\
+----------+--------+------------+------------+---------------------+-------------------+
| callsign | icao24 | firstseen  | lastseen   | estdepartureairport | estarrivalairport |
+----------+--------+------------+------------+---------------------+-------------------+
| SKW5466  | 1676665124 | 1676667470 | KBTR                | NULL              |
+----------+--------+------------+------------+---------------------+-------------------+\
"""

    output_bad_5 = pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid input, data line 0 does not agree with columns \
format\n| SKW5466  | 1676665124 | 1676667470 | KBTR                \
| NULL              |"
        ),
    )

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


def test_to_unix_timestamp():
    """
    Test the `to_unix_timestamp` function from `src.backend.utils`.

    The function is tested for:
    - Conversion of date strings in various formats to UNIX timestamps.
    - Conversion of UNIX timestamp strings to integers.
    - Conversion of UNIX timestamp integers and floats.
    - Handling of date and datetime objects.
    - Handling of unsupported date formats and `None` inputs.
    """
    # ----------------Defining Good and Bad Cases---------------- #
    input_good_1 = "2022-01-01"

    input_good_2 = "2022-01-01 00:00:00"

    input_good_3 = "1640995200"

    input_good_4 = 1640995200

    input_good_5 = 1640995200.0

    input_good_5 = datetime.date(2022, 1, 1)

    input_good_6 = datetime.datetime(2022, 1, 1, 0, 0)

    output_good = 1640995200

    input_bad_1 = "a"

    input_bad_2 = None

    output_bad = pytest.raises(ValueError, match=re.escape("Unsupported date format"))

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
