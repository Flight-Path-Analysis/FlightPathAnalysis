"""
The `compressors` module provides a suite of tools for compressing,
encoding, and decoding data using B-splines.

Primary functionalities include:
1. Compressing and encoding data from a given dataframe into a more compact
   representation.
2. Computing the optimal smoothing factor for splines to achieve a desired
   uncertainty.
3. Calculating the uncertainty of the spline-reconstructed data against the
   original dataset.
4. Converting the spline information into a metadata dictionary for easier
   storage or transmission and vice-versa.
5. Computing the compression factor to evaluate the efficiency of the
   compression.
6. Saving the encoded spline data to a YAML file for persistence.

Classes:
- SplineCompressor:
    Provides methods for data compression using B-splines. This includes
    determining the optimal smoothing factor, computing uncertainty,
    encoding to metadata, and more.
- ScaledSpline:
    A wrapper around a BSpline object that incorporates scaling functionalities.
    It takes into account the original range of the data to provide descaled
    outputs.

Dependencies:
- External Libraries: yaml, numpy, scipy, IPython, pympler
- Internal Modules: src.backend.utils

Note:
Ensure the necessary configurations and dependencies are correctly set up
before using functionalities from this module.
"""

import yaml
from scipy.interpolate import UnivariateSpline, BSpline
import numpy as np
from pympler import asizeof
import pandas as pd
import os
from src.backend import utils


class SplineCompressor:
    """
    SplineCompressor offers utilities to compress and encode data using B-splines.

    It provides methods to determine the optimal smoothing factor for splines,
    compute the uncertainty of the spline-reconstructed data against the original data
    and encode and decode splines for more compact storage or transmission.

    Attributes:
    - max_error (float): Threshold for acceptable range of uncertainty calculation.
    - s_min_precision (float): Minimum acceptable precision for the smoothing factor 's'.
    - s_base_precision (float): Starting precision for the smoothing factor 's'.
    - logger (utils.Logger, optional): Logging utility.

    Methods:
    - best_spline_s(xs, ys): Determine optimal 's' for a UnivariateSpline.
    - compute_spline_uncertainty(xs, ys, spline):
        Compute uncertainty if the spline data against the original.
    - encode_spline(spline, metadata={}): Convert spline to metadata dictionary.
    - optimize_and_encode_spline(xs, ys, metadata={}): Optimize spline and convert to metadata.
    - compute_compression_factor(xs, ys, metadata): Compute compression factor of encoding.
    - decode_spline(metadata): Decode a BSpline from metadata.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the SplineCompressor object.

        Parameters:
        - config (dict): Configuration settings for the SplineCompressor.
        - logger (utils.Logger): Logging utility for the compressor (default is None).
        """
        self.config = config
        self.degree = config["data-compression"]["spline-compressor"]["degree"]
        self.max_error = config["data-compression"]["spline-compressor"]["max-error"]
        self.s_min_precision = config["data-compression"]["spline-compressor"][
            "s-minimum-precision"
        ]
        self.s_base_precision = config["data-compression"]["spline-compressor"][
            "s-base-precision"
        ]
        self.logger = logger

    def treat_angular_data(self, heading_y, thresh=330):
        """
        Treats angular data by adjusting any sudden jumps in
        heading_y values greater than the threshold.

        Parameters:
        heading_y (list): A list of heading values in degrees.
        thresh (int): The threshold value for sudden jumps in heading_y values. Defaults to 330.

        Returns:
        list: The treated list of heading values.
        """
        for i, curr_y in enumerate(heading_y[1:], 1):
            prev_y = heading_y[i - 1]
            if curr_y - prev_y > thresh:
                heading_y[i] = curr_y - 360
            if prev_y - curr_y > thresh:
                heading_y[i] = curr_y + 360
        return heading_y

    def best_spline_s(self, xs, ys):
        """
        Determine the best smoothing factor 's' for a UnivariateSpline that
        achieves a desired uncertainty while attempting to minimize the number
        of spline coefficients and knots.

        Parameters:
        - xs : array-like
            Independent variable data.
        - ys : array-like
            Dependent variable data corresponding to 'xs'.

        Returns:
        - float
            The determined optimal smoothing factor 's'.
        """

        # Initialized the precision and s before narrowing down on the first loop
        s_precision = self.s_base_precision
        s = 0

        # Increase 's' until we achieve the desired uncertainty or until s_precision is too small
        while s_precision > self.s_min_precision:
            s += s_precision
            spline = UnivariateSpline(xs, ys, s=s, k=self.degree)
            err = self.compute_spline_uncertainty(xs, ys, spline)
            # print(err, self.max_error, s, s_precision)
            if err > self.max_error:
                s -= s_precision
                s_precision /= 2

        # Compute L for the best s so far
        spline = UnivariateSpline(xs, ys, s=s, k=self.degree)

        # Get the number of coefficients and knots for current 's'
        complexity = len(spline.get_coeffs()) + len(spline.get_knots())

        # Attempt to reduce 's' without increasing the complexity (number of coefficients and knots)
        s_precision = s / 2
        while s_precision > self.s_min_precision:
            s -= s_precision
            if s >= 0:
                spline = UnivariateSpline(xs, ys, s=s, k=self.degree)
                l = len(spline.get_coeffs()) + len(spline.get_knots())
            else:
                l = np.inf
            if l > complexity:
                s += s_precision
                s_precision /= 2
        return s

    def compute_spline_uncertainty(self, xs, ys, spline):
        """
        Compute the uncertainty of the spline-reconstructed data against the original
        data's ys and the ys.

        Parameters:
        - xs (array-like): Independent variable data.
        - ys (array-like): Original dependent variable data.
        - spline (UnivariateSpline): Spline function to be evaluated.

        Returns:
        - float: Uncertainty measure.
        """
        spline_ys = spline(xs)
        sigma = np.sqrt(np.mean((ys - spline_ys) ** 2))
        return sigma

    def encode_spline(self, spline, metadata=None):
        """
        Convert a spline function into a metadata dictionary for easier storage or transmission.

        Parameters:
        - spline (UnivariateSpline): The spline to encode.
        - metadata (dict, optional): Any additional metadata to include.

        Returns:
        - dict: Metadata representing the spline.
        """
        if not metadata:
            metadata = {}
        coeffs = spline.get_coeffs().tolist()
        knots = spline.get_knots().tolist()
        metadata.update({"coefficients": coeffs, "knots": knots})
        return metadata

    def optimize_and_encode_spline(self, xs, ys, metadata=None):
        """
        Optimize the spline based on xs and ys, then encode the spline into metadata.

        Parameters:
        - xs (array-like): Independent variable data.
        - ys (array-like): Dependent variable data corresponding to 'xs'.
        - metadata (dict): Dictionary to populate with encoded spline data.

        Returns:
        - dict: Metadata dictionary containing encoded spline and other info.
        """
        if not metadata:
            metadata = {}
        best_s = self.best_spline_s(xs, ys)
        spline = UnivariateSpline(xs, ys, s=best_s, k=self.degree)
        metadata = self.encode_spline(spline, metadata=metadata)
        return metadata

    def compute_compression_factor(self, xs, ys, metadata):
        """
        Compute the compression factor achieved by encoding xs and ys into the given metadata.

        Parameters:
        - xs (array-like): Original independent variable data.
        - ys (array-like): Original dependent variable data.
        - metadata (dict): Encoded metadata representation of xs and ys.

        Returns:
        - float: Compression factor.
        """
        original = {"xs": list(xs.astype(float)), "ys": list(ys)}
        size_original = asizeof.asizeof(original)
        size_new = asizeof.asizeof(metadata)
        return size_original / size_new

    def decode_spline_single_column(self, metadata, column):
        """
        Decode a BSpline from given metadata and return a function that outputs
        values in the original range.

        Parameters:
        - metadata (dict): Metadata containing encoded spline info for the column.

        Returns:
        - function: A function that takes scaled x-values and returns y-values
        in the original range.
        """

        def newknots(knots, k):
            """
            Adjust the knots for the boundary of order k.

            Parameters:
            - knots (list): List of original knots.
            - k (int): Boundary order.

            Returns:
            - list: Adjusted knots list.
            """
            return [knots[0]] * k + list(knots) + [knots[-1]] * k

        # Ensure the necessary metadata keys are present
        if column not in metadata:
            raise ValueError(f"Missing key {column} in metadata.")

        required_keys = ["xmin", "xmax", "ymin", "ymax"]
        for key in required_keys:
            if key not in metadata[column]:
                raise ValueError(f"Missing key {key} in metadata of {column}.")

        x_min = metadata[column]["xmin"]
        x_max = metadata[column]["xmax"]
        y_min = metadata[column]["ymin"]
        y_max = metadata[column]["ymax"]

        knots = metadata[column]["knots"]
        coeffs = metadata[column]["coefficients"]

        # Get boundary knots for scipy's BSpline
        forscipyknots = newknots(knots, self.degree)
        spline = BSpline(forscipyknots, coeffs, self.degree)

        # Return an instance of ScaledSpline
        return self.ScaledSpline(
            spline, {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        )

    def encode_from_dataframe(
        self, dataframe, independent_variable, dependent_variable, metadata=None
    ):
        """
        Encode data from a dataframe based on the provided independent and dependent variables.

        Parameters:
        - dataframe (pd.DataFrame): The data source.
        - independent_variable (str): Column name for the independent variable.
        - dependent_variable (str/list[str]): Column name(s) for the dependent variable(s).
        - metadata (dict, optional): Any additional metadata to include. Default is an empty dict.

        Returns:
        - ScaledSpline: An instance of the ScaledSpline class.

        Raises:
        - ValueError: If the types of provided independent or dependent variables
        are not as expected.
        """
        if not metadata:
            metadata = {}

        # Ensure independent variable is a string
        if not isinstance(independent_variable, str):
            raise ValueError(
                f"Expected independent_variable to be a string, \
                got {type(independent_variable)} instead."
            )
        dataframe["lat"] = self.treat_angular_data(dataframe["lat"].to_list())
        dataframe["heading"] = self.treat_angular_data(dataframe["heading"].to_list())
        # Store the name of the independent variable and some run parameters in the metadata
        metadata["x_variable"] = independent_variable
        metadata["max_error"] = self.max_error
        metadata["s_min_precision"] = self.s_min_precision

        # Normalize and scale the independent variable data
        x_min = min(dataframe[independent_variable].values)
        x_max = max(dataframe[independent_variable].values)
        xs = (dataframe[independent_variable].values - x_min) / (x_max - x_min)

        original_xs_new = np.arange(
            dataframe[independent_variable].values[0],
            dataframe[independent_variable].values[-1] + 1)
        xs_new = (original_xs_new - x_min) / (x_max - x_min)

        # Convert the dependent variable to a list format if it's a single string
        if isinstance(dependent_variable, str):
            dependent_variable = [dependent_variable]
        # Check if all elements in the dependent variable list are strings
        elif not all(isinstance(y_var, str) for y_var in dependent_variable):
            raise ValueError("All items in dependent_variable should be strings.")

        metadata["y_variables"] = dependent_variable
        metadata["compressor"] = "SplineCompressor"
        # Loop through each dependent variable for encoding
        for y_variable in dependent_variable:
            # Ensure current dependent variable is a string
            if not isinstance(y_variable, str):
                raise ValueError(
                    f"Expected y_variable to be a string, got {type(y_variable)} instead."
                )

            # Normalize and scale the dependent variable data
            y_min = min(dataframe[y_variable].values)
            y_max = max(dataframe[y_variable].values)
            ys_new = np.interp(xs_new, xs, (dataframe[y_variable].values - y_min) / (y_max - y_min))

            # Store min-max details in the metadata
            metadata[y_variable] = {
                "xmin": float(x_min),
                "xmax": float(x_max),
                "ymin": float(y_min),
                "ymax": float(y_max),
                "best_s": float(self.best_spline_s(xs_new, ys_new)),
            }

            # Optimize and encode the spline, updating the metadata for the
            # current dependent variable
            metadata[y_variable] = self.optimize_and_encode_spline(
                xs_new, ys_new, metadata=metadata[y_variable]
            )

        return metadata

    def encode_from_dataframe_to_file(
        self, dataframe, independent_variable, dependent_variable, metadata=None
    ):
        """
        Encode the data from a dataframe into a metadata dictionary based on
        the provided independent and dependent variables, and then save the
        metadata to a YAML file.

        Parameters:
        - dataframe (pd.DataFrame): The data source.
        - independent_variable (str): Column name for the independent variable.
        - dependent_variable (str/list[str]): Column name(s) for the dependent variable(s).
        - filename (str, optional): Name of the output file to save the metadata.
        If not provided, it will be determined from the metadata.
        - metadata (dict, optional): Any additional metadata to include. Default is an empty dict.

        The method uses the `flight_id` from the metadata to determine the filename if not provided.
        The path of the file is determined using the 'base-configs' and
        'data-gather' configurations.
        """
        if not metadata:
            metadata = {}
        # Encode the data from the dataframe into a metadata dictionary
        metadata = self.encode_from_dataframe(
            dataframe, independent_variable, dependent_variable, metadata=metadata
        )

        flight_id = metadata["flight_id"]
        filename = f"{flight_id}.yml"

        # Convert the metadata dictionary to a YAML formatted string
        yaml_data = yaml.dump(metadata, default_flow_style=None)

        # Retrieve directory configurations to build the full file path
        basedir = utils.clean_path(self.config["base-configs"]["root-directory"])
        outdir = utils.clean_path(self.config["data-gather"]["flights"]["out-dir"])

        # Build the full file path
        filename = os.path.join(basedir, outdir, filename)

        # Write the YAML data to the file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(yaml_data)

    class ScaledSpline:
        """
        ScaledSpline wraps around a BSpline object to provide scaling and descaling functionalities.
        Parameters:
        - spline (BSpline): The spline object.
        - x_min (float): Minimum x-value of the original data.
        - x_max (float): Maximum x-value of the original data.
        - y_min (float): Minimum y-value of the original data.
        - y_max (float): Maximum y-value of the original data.
        """

        def __init__(self, spline, ranges):
            self.spline = spline
            self.x_min = ranges["x_min"]
            self.x_max = ranges["x_max"]
            self.y_min = ranges["y_min"]
            self.y_max = ranges["y_max"]

        def __call__(self, x):
            """
            Call the ScaledSpline object with x-values.
            Parameters:
            - x (array-like): Input x-values.
            Returns:
            - array-like: Corresponding y-values in the original data range.
            """
            # Ensure the x-values are within the range
            if np.any(x < self.x_min) or np.any(x > self.x_max):
                raise ValueError(
                    f"Input x-values should be within the range [{self.x_min}, {self.x_max}]"
                )

            # Scale the x-values to [0, 1] range
            x_scaled = (x - self.x_min) / (self.x_max - self.x_min)

            # Get the y-values from the original spline
            y_scaled = self.spline(x_scaled)

            # Descale the y-values to original range
            y_original = y_scaled * (self.y_max - self.y_min) + self.y_min

            return y_original

        def get_ranges(self):
            """
            Return the original range of x and y values.

            Returns:
            - dict: A dictionary containing the min and max values for x and y.
            """
            return {
                "xmin": float(self.x_min),
                "xmax": float(self.x_max),
                "ymin": float(self.y_min),
                "ymax": float(self.y_max),
            }


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

        return pd.DataFrame(df_new)
