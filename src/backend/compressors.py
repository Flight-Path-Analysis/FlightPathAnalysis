from scipy.interpolate import UnivariateSpline
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import pearsonr
from IPython.display import clear_output
from pympler import asizeof
import sys
sys.path.append("../../")
from src.backend import utils
import yaml

class SplineCompressor:
    """
    SplineCompressor offers utilities to compress and encode data using B-splines.

    It provides methods to determine the optimal smoothing factor for splines,
    compute Pearson correlation coefficients for the original data and the spline-reconstructed data,
    and encode and decode splines for more compact storage or transmission.

    Attributes:
    - pearson_corr_range_threshold (float): Threshold for acceptable range of Pearson correlation.
    - s_min_precision (float): Minimum acceptable precision for the smoothing factor 's'.
    - s_base_precision (float): Starting precision for the smoothing factor 's'.
    - logger (utils.Logger, optional): Logging utility.

    Methods:
    - best_spline_s(xs, ys): Determine optimal 's' for a UnivariateSpline.
    - compute_spline_pearson_corr(xs, ys, spline): Compute Pearson correlation between original and spline data.
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
        self.pearson_corr_range_threshold = config['data-compression']['spline-compressor']['pearson_corr_range_threshold']
        self.s_min_precision = config['data-compression']['spline-compressor']['s_minimum_precision']
        self.s_base_precision = config['data-compression']['spline-compressor']['s_base_precision']
        self.logger = logger
    
    def best_spline_s(self, xs, ys):
        """
        Determine the best smoothing factor 's' for a UnivariateSpline that achieves a desired Pearson correlation
        while attempting to minimize the number of spline coefficients and knots.

        Parameters:
        - xs : array-like
            Independent variable data.
        - ys : array-like
            Dependent variable data corresponding to 'xs'.

        Returns:
        - float
            The determined optimal smoothing factor 's'.
        """
        
        #Initialized the precision and s before narrowing down on the first loop
        s_precision = self.s_base_precision
        s = 0

        # Calculate correlation for the worst-case (no smoothing)
        worst_spline = UnivariateSpline(xs, ys, s=np.inf)
        worst_corr = self.compute_spline_pearson_corr(xs, ys, worst_spline)

        # Set correlation threshold based on the worst case
        corr_thresh = worst_corr + (1 - worst_corr) * self.pearson_corr_range_threshold


        # Increase 's' until we achieve the desired correlation or until s_precision is too small
        while s_precision > self.s_min_precision:        
            s += s_precision
            spline = UnivariateSpline(xs, ys, s=s)
            corr = self.compute_spline_pearson_corr(xs, ys, spline)
            clear_output(wait = True)

            if corr < corr_thresh:
                s -= s_precision
                s_precision /= 2

        # Compute L for the best s so far
        spline = UnivariateSpline(xs, ys, s=s)

        # Get the number of coefficients and knots for current 's'
        L = len(spline.get_coeffs()) + len(spline.get_knots())

        # Attempt to reduce 's' without increasing the complexity (number of coefficients and knots)
        s_precision = s/2
        while s_precision > self.s_min_precision:
            s -= s_precision
            if s >= 0:
                spline = UnivariateSpline(xs, ys, s = s)
                l = len(spline.get_coeffs()) + len(spline.get_knots())
            else:
                l = np.inf
            clear_output(wait = True)
            if l > L:
                s += s_precision
                s_precision /= 2
        return s
    
    def compute_spline_pearson_corr(self, xs, ys, spline):
        """
        Compute the Pearson correlation coefficient for the original ys and the ys computed using the provided spline.

        Parameters:
        - xs (array-like): Independent variable data.
        - ys (array-like): Original dependent variable data.
        - spline (UnivariateSpline): Spline function to be evaluated.

        Returns:
        - float: Pearson correlation coefficient.
        """
        spline_ys = spline(xs)
        return pearsonr(ys, spline_ys)[0]
    
    def encode_spline(self, spline, metadata={}):
        """
        Convert a spline function into a metadata dictionary for easier storage or transmission.

        Parameters:
        - spline (UnivariateSpline): The spline to encode.
        - metadata (dict, optional): Any additional metadata to include.

        Returns:
        - dict: Metadata representing the spline.
        """
        coeffs = spline.get_coeffs().tolist()
        knots = spline.get_knots().tolist()
        metadata.update({
            'coefficients': coeffs,
            'knots': knots
        })
        return metadata
        
    def optimize_and_encode_spline(self, xs, ys, metadata = {}):
        """
        Optimize the spline based on xs and ys, then encode the spline into metadata.

        Parameters:
        - xs (array-like): Independent variable data.
        - ys (array-like): Dependent variable data corresponding to 'xs'.
        - metadata (dict): Dictionary to populate with encoded spline data.

        Returns:
        - dict: Metadata dictionary containing encoded spline and other info.
        """
        best_s = self.best_spline_s(xs, ys)
        spline = UnivariateSpline(xs, ys, s=best_s)
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
        original = {'xs': list(xs.astype(float)), 'ys': list(ys)}
        size_original = asizeof.asizeof(original)
        size_new = asizeof.asizeof(metadata)
        return size_original / size_new

    def decode_spline_single_column(self, metadata, column):
        """
        Decode a BSpline from given metadata and return a function that outputs values in the original range.

        Parameters:
        - metadata (dict): Metadata containing encoded spline info for the column.

        Returns:
        - function: A function that takes scaled x-values and returns y-values in the original range.
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
            raise ValueError(f"Missing key {key} in metadata.")
            
        required_keys = ['xmin', 'xmax', 'ymin', 'ymax']
        for key in required_keys:
            if key not in metadata[column]:
                raise ValueError(f"Missing key {key} in metadata of {column}.")
                
        minX = metadata[column]['xmin']
        maxX = metadata[column]['xmax']
        minY = metadata[column]['ymin']
        maxY = metadata[column]['ymax']

        knots = metadata[column]['knots']
        coeffs = metadata[column]['coefficients']

        # Get boundary knots for scipy's BSpline
        forscipyknots = newknots(knots, 3)
        spline = BSpline(forscipyknots, coeffs, 3)

        # Return an instance of ScaledSpline
        return self.ScaledSpline(spline, minX, maxX, minY, maxY)

    
    def encode_from_dataframe(self, dataframe, independent_variable, dependent_variable, metadata={}):
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
        - ValueError: If the types of provided independent or dependent variables are not as expected.
        """
        if metadata is None:
            metadata = {}

        # Ensure independent variable is a string
        if not isinstance(independent_variable, str):
            raise ValueError(f"Expected independent_variable to be a string, got {type(independent_variable)} instead.")

        # Store the name of the independent variable and some run parameters in the metadata
        metadata['x_variable'] = independent_variable
        metadata['pearson_corr_range_threshold'] = self.pearson_corr_range_threshold
        metadata['s_min_precision'] = self.s_min_precision
        
        # Normalize and scale the independent variable data
        original_xs = dataframe[independent_variable].values
        minX = min(original_xs)
        maxX = max(original_xs)
        xs = (original_xs - minX) / (maxX - minX)

        # Convert the dependent variable to a list format if it's a single string
        if isinstance(dependent_variable, str):
            dependent_variable = [dependent_variable]
        # Check if all elements in the dependent variable list are strings
        elif not all(isinstance(y_var, str) for y_var in dependent_variable):
            raise ValueError("All items in dependent_variable should be strings.")
        
        metadata['y_variables'] = dependent_variable
        # Loop through each dependent variable for encoding
        for y_variable in dependent_variable:

            # Ensure current dependent variable is a string
            if not isinstance(y_variable, str):
                raise ValueError(f"Expected y_variable to be a string, got {type(y_variable)} instead.")

            # Normalize and scale the dependent variable data
            original_ys = dataframe[y_variable].values
            minY = min(original_ys)
            maxY = max(original_ys) 
            ys = (original_ys - minY) / (maxY - minY)

            # Compute the best smoothing factor 's' for the spline
            best_s = self.best_spline_s(xs, ys)
            spline = UnivariateSpline(xs, ys, s=best_s)

            # Store min-max details in the metadata
            metadata[y_variable] = {
                    'xmin': minX,
                    'xmax': maxX,
                    'ymin': minY,
                    'ymax': maxY}

            # Optimize and encode the spline, updating the metadata for the current dependent variable
            metadata[y_variable] = self.optimize_and_encode_spline(xs, ys, metadata=metadata[y_variable])

        return metadata

    def encode_from_dataframe_to_file(self, dataframe, independent_variable, dependent_variable, filename=None, metadata={}):
        """
        Encode the data from a dataframe into a metadata dictionary based on the provided independent and dependent variables,
        and then save the metadata to a YAML file.

        Parameters:
        - dataframe (pd.DataFrame): The data source.
        - independent_variable (str): Column name for the independent variable.
        - dependent_variable (str/list[str]): Column name(s) for the dependent variable(s).
        - filename (str, optional): Name of the output file to save the metadata. If not provided, it will be determined from the metadata.
        - metadata (dict, optional): Any additional metadata to include. Default is an empty dict.

        The method uses the `flight_id` from the metadata to determine the filename if not provided.
        The path of the file is determined using the 'base-configs' and 'data-gather' configurations.
        """
        
        # Encode the data from the dataframe into a metadata dictionary
        metadata = self.encode_from_dataframe(dataframe, independent_variable, dependent_variable, metadata = metadata)
        
        # If no filename is provided, use the 'flight_id' from the metadata to create one
        if not filename:
            flight_id = metadata['flight_id']
            filename = f'{flight_id}.yml'
        
        # Convert the metadata dictionary to a YAML formatted string
        yaml_data = yaml.dump(metadata, default_flow_style=None)
        
        # Retrieve directory configurations to build the full file path
        basedir = self.config['base-configs']['root-directory']
        outdir = self.config['data-gather']['flights']['out-dir']
        
        # Ensure directories end with a '/'
        if not basedir.endswith('/'):
            basedir += '/'
        if not outdir.endswith('/'):
            outdir += '/'
            
        # Build the full file path
        filename = basedir + outdir + filename
        
        # Write the YAML data to the file
        with open(filename, 'w') as f:
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
        def __init__(self, spline, x_min, x_max, y_min, y_max):
            self.spline = spline
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

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
                raise ValueError(f"Input x-values should be within the range [{self.x_min}, {self.x_max}]")

            # Scale the x-values to [0, 1] range
            x_scaled = (x - self.x_min) / (self.x_max - self.x_min)

            # Get the y-values from the original spline
            y_scaled = self.spline(x_scaled)

            # Descale the y-values to original range
            y_original = y_scaled * (self.y_max - self.y_min) + self.y_min

            return y_original