from scipy.interpolate import UnivariateSpline
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import pearsonr
from IPython.display import clear_output
from pympler import asizeof
import utils

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
        coeffs = list(spline.get_coeffs())
        knots = list(spline.get_knots())
        metadata.update({
            'pearson_corr_range_threshold': self.pearson_corr_range_threshold,
            's_min_precision': self.s_min_precision,
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
        x0 = xs[0]
        xf = xs[-1]
        metadata['x0'] = x0
        metadata['xf'] = xf
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

    def decode_spline(self, metadata):
        """
        Decode a BSpline from given metadata.

        Parameters:
        - metadata (dict): Metadata containing encoded spline info.

        Returns:
        - BSpline: Decoded spline object.
        """
        knots = metadata['knots']
        coeffs = metadata['coefficients']
        
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
        
        # Get boundary knots for scipy's BSpline
        forscipyknots = newknots(knots, 3)
        return BSpline(forscipyknots, coeffs, 3)
    