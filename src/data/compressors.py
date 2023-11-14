import numpy as np
from pympler import asizeof
import pandas as pd

class CsvCompressor:
    """
    A class for compressing and decompressing CSV files.

    The CsvCompressor class provides functionalities to encode dataframes
    into CSV files and decode them back into dataframes. It supports
    reindexing and interpolating missing values in the 'time' column
    for sequential data representation.

    Attributes:
    - config (dict): A dictionary containing configuration information.
    - logger (Logger, optional): An optional logger object to log messages.

    Methods:
    - log_verbose(message): Logs a provided message if a logger is configured.
    - encode_from_dataframe_to_file(dataframe, flight_id): Encodes a dataframe into a CSV file.
    - decode_to_dataframe_from_file(filename): Decodes a CSV file back 
    into a dataframe with interpolated values.
    """

    def __init__(self, config, logger=None):
        """
        Initialize a CsvCompressor object.

        This method initializes a CsvCompressor object with the provided configuration
        and logger (if any).

        Parameters:
        - config (dict): A dictionary containing configuration information.
        - logger (Logger, optional): A logger object to be used for logging messages.
        """
        self.config = config
        self.logger = logger

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

    def encode_from_dataframe_to_file(self, dataframe, filename):
        """
        Encode a dataframe to a CSV file.

        This method encodes a provided dataframe to a CSV file with the provided metadata
        and flight ID.

        Parameters:
        - dataframe (pandas.DataFrame): The dataframe to be encoded.
        - flight_id (str): The ID of the flight being encoded.

        Returns:
        - compression_ratio (float): The compression ratio achieved.
        """
        columns = [
            "time",
            "lat",
            "lon",
            "baroaltitude",
            "geoaltitude",
            "heading",
            "velocity",
        ]
        self.log_verbose(f"CSV Encoding data for {str(filename).split('/')[-1]}")

        df_interp = {
            "time": np.linspace(
                dataframe["time"].iloc[0],
                dataframe["time"].iloc[-1],
                num=self.config["data-compression"]["csv-compressor"]["num-points"],
                endpoint=True,
            )
        }
        for col in columns[1:]:
            df_interp[col] = np.interp(
                df_interp["time"], dataframe["time"], dataframe[col]
            )
            if col in ("lat", "heading"):
                df_interp[col] = np.mod(df_interp[col], 360)
        df_interp = pd.DataFrame(df_interp)
        temp_csv = f"{filename}"
        df_interp[columns].to_csv(temp_csv)

        # Calculate and return compression ratio
        compression_ratio = asizeof.asizeof(dataframe) / asizeof.asizeof(df_interp)

        self.log_verbose(f"Compression ratio achieved: {compression_ratio}")

    def decode_to_dataframe_from_file(self, filename):
        """
        Decode a CSV file to a dataframe.

        This method decodes a provided CSV file to a dataframe, separating metadata
        and data rows.

        Parameters:
        - filename (str): The name of the file to be decoded.

        Returns:
        - dataframe (pandas.DataFrame): The decoded dataframe.
        """
        columns = [
            "time",
            "lat",
            "lon",
            "baroaltitude",
            "geoaltitude",
            "heading",
            "velocity",
        ]

        self.log_verbose(f"CSV Decoding data from {filename}")

        # Read the CSV, separating metadata and dataframe
        dataframe = pd.read_csv(filename, index_col=0)[columns]
        df_new = {
            "time": list(
                range(
                    int(dataframe["time"].iloc[0]), int(dataframe["time"].iloc[-1]) + 1
                )
            )
        }
        for col in columns[1:]:
            df_new[col] = np.interp(df_new["time"], dataframe["time"], dataframe[col])
            if col in ("lat", "heading"):
                df_new[col] = np.mod(df_new[col], 360)
        df_new = pd.DataFrame(df_new)
        cols_to_check = ['lat', 'lon']
        df_new = df_new.drop_duplicates(subset=cols_to_check, keep='first')
        return pd.DataFrame(df_new)
