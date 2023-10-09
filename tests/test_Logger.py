import sys
sys.path.append("../")
from src.backend.utils import Logger
import os
import datetime
import pytest

@pytest.fixture
def mock_logger():
    config = {
        'base-configs': {
            'root-directory': './test_dir',
            'tag': 'test_tag'
        },
        'log': {
            'log-directory': 'logs'
        }
    }
    return Logger(config)

def test_clean_path_with_invalid_input(mock_logger):
    with pytest.raises(ValueError, match='Path must be a string'):
        mock_logger.clean_path(123)  # Passing integer, should raise error

def test_log_with_invalid_input(mock_logger):
    with pytest.raises(ValueError, match='Text to log must be a string'):
        mock_logger.log(456)  # Passing integer, should raise error

def test_clean_path(mock_logger):
    path = './test_dir'
    cleaned_path = mock_logger.clean_path(path)
    assert cleaned_path == './test_dir/'

def test_log_creates_file_and_directory(mock_logger):
    # Removing directory if it exists from previous test run
    if os.path.exists('./test_dir/logs'):
        os.rmdir('./test_dir/logs')
    if os.path.exists('./test_dir'):
        os.rmdir('./test_dir')

    mock_logger.log('Test message')
    assert os.path.exists('./test_dir/logs/test_tag.log')
    
def test_log_content(mock_logger):
    test_message = 'Another test message'
    mock_logger.log(test_message)
    with open('./test_dir/logs/test_tag.log', 'r') as file:
        lines = file.readlines()
        # Checking the last line since we appended to the log
        last_line = lines[-1]
        assert test_message in last_line

        # Ensure the datetime was correctly formatted
        current_date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M')
        assert current_date in last_line

    # Cleaning up
    os.remove('./test_dir/logs/test_tag.log')
    os.rmdir('./test_dir/logs')
    os.rmdir('./test_dir')
