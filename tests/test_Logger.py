"""
Unit tests for the Logger class from src.backend.utils.

This module contains tests that validate the behavior of the Logger class methods,
ensuring correct path manipulation, file creation, and content logging.
"""

# pylint: disable=redefined-outer-name

import os
import datetime
import pytest

from src.backend.utils import Logger

@pytest.fixture
def mock_logger():
    """
    Pytest fixture that provides a mock Logger instance for testing.

    Returns:
        Logger: A Logger instance with predefined configuration.
    """
    config = {
        "base-configs": {"root-directory": "./test_dir", "tag": "test_tag"},
        "log": {"log-directory": "logs"},
    }
    return Logger(config)


def test_log_with_invalid_input(mock_logger):
    """
    Test the numerical input behavior of the Logger class.
    """
    with pytest.raises(ValueError, match="Text to log must be a string"):
        mock_logger.log(456)


def test_clean_path(mock_logger):
    """
    Test the path behavior of the Logger class.
    """
    path = "./test_dir"
    cleaned_path = mock_logger.clean_path(path)
    assert cleaned_path == "./test_dir/"


def test_log_creates_file_and_directory(mock_logger):
    """
    Test the file creation behavior of the Logger class.
    """

    # Removing directory if it exists from previous test run
    if os.path.exists("./test_dir/logs"):
        os.rmdir("./test_dir/logs")
    if os.path.exists("./test_dir"):
        os.rmdir("./test_dir")

    mock_logger.log("Test message")
    assert os.path.exists("./test_dir/logs/test_tag.log")


def test_log_content(mock_logger):
    """
    Test the file content behavior of the Logger class.
    """
    test_message = "Another test message"
    mock_logger.log(test_message)
    with open("./test_dir/logs/test_tag.log", "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Checking the last line since we appended to the log
        last_line = lines[-1]
        assert test_message in last_line

        # Ensure the datetime was correctly formatted
        current_date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
        assert current_date in last_line

    # Cleaning up
    os.remove("./test_dir/logs/test_tag.log")
    os.rmdir("./test_dir/logs")
    os.rmdir("./test_dir")
