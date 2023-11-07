import numpy as np
from scipy.interpolate import interp1d


def calibrate_stations(stations_data, config):
    """
    Calibrate multiple station data by fitting temperature and wind models and predict sea-level values.

    This function iterates through each station's data, fits temperature and wind speed models based on the 
    available data, and then uses these models to predict certain values at sea level. These predictions are 
    then added to the original data frame.

    Parameters:
    stations_data (pd.DataFrame): A DataFrame containing the weather data for multiple stations. Each row 
                                  should correspond to a different station and contain columns for various 
                                  weather parameters, such as temperature, wind speed, and elevation.
                                  Expected columns are 'elevation', 'tmpf', 'sknt_E', and 'sknt_N'. 

    config (dict): A dictionary containing the configuration settings for the temperature and wind models, 
                   including the type of model and any necessary default coefficients.

    Returns:
    pd.DataFrame: The original stations_data DataFrame is returned, with additional columns for the models and 
                  sea-level predictions. New columns will be 'temperature_model', 'wind_model_E', 'wind_model_N', 
                  'tmpf_sea_level', 'sknt_E_sea_level', and 'sknt_N_sea_level'.
    """
    # Prepare a container for the models. Each station will have its own model.
    temperature_models = [TemperatureModel(config) for _ in range(len(stations_data))]
    wind_E_models = [WindModel(config) for _ in range(len(stations_data))]
    wind_N_models = [WindModel(config) for _ in range(len(stations_data))]
    air_pressure_models = [AirPressureModel(config) for _ in range(len(stations_data))]
    air_density_models = [AirDensityeModel(config) for _ in range(len(stations_data))]
    cloud_models = [CloudModel(config) for _ in range(len(stations_data))]

    # Iterate over stations data and fit models based on the data for each station.
    for i, row in stations_data.iterrows():
        elevations = [row['elevation']]
        temperatures = [row['tmpf']]
        clouds = row['skyc1', 'skyc2', 'skyc3', 'skyc4'].values.tolist()
        cloud_levels = row['skyl1', 'skyl2', 'skyl3', 'skyl4'].values.tolist()
        wind_speeds_E = [row['sknt_E']]
        wind_speeds_N = [row['sknt_N']]

        # Fit the model for this station.
        temperature_models[i].fit(elevations, temperatures)
        wind_E_models[i].fit(elevations, wind_speeds_E)
        wind_N_models[i].fit(elevations, wind_speeds_N)
        air_pressure_models[i].fit(temperature_models[i])
        air_density_models[i].fit(air_pressure_models[i], temperature_models[i])
        cloud_models[i].fit(cloud_levels, clouds)

    # Assign models to a new column in the DataFrame.
    stations_data['tmpf_model'] = temperature_models
    stations_data['sknt_E_model'] = wind_E_models
    stations_data['sknt_N_model'] = wind_N_models
    stations_data['air_pressure_model'] = air_pressure_models
    stations_data['air_density_model'] = air_density_models
    stations_data['cloud_model'] = cloud_models
    
    # Predict the sea-level temperature for each station using its respective model.
    stations_data['tmpf_sea_level'] = [row['tmpf_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['sknt_E_sea_level'] = [row['sknt_E_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['sknt_N_sea_level'] = [row['sknt_E_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['air_pressure_sea_level'] = [row['air_pressure_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['air_density_sea_level'] = [row['air_density_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['cloud_sea_level'] = [row['cloud_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    
    return stations_data

class TemperatureModel:
    """
    Represents a temperature model, typically linear, that can be fitted to data points and used for predictions.

    Attributes:
        config (dict): Configuration dictionary containing model parameters.
        model (str): Type of model to be used, retrieved from the configuration.
        coeffs (list): Coefficients of the fitted model.
        fitter (bool): Flag indicating if the model has been fitted (not actively used in the current implementation).
    """

    def __init__(self, config):
        """Initialize the TemperatureModel with the given configuration."""
        self.config = config
        self.model = config['models']['temperature']['model']
        self.coeffs = []
        self.fitted = False

    def fit(self, heights, temperatures):
        """
        Fit the model based on the provided heights and temperatures.

        Args:
            heights (list): List of height measurements.
            temperatures (list): List of corresponding temperature measurements.

        Raises:
            ValueError: If the model type is unknown or unsupported.
        """
        if self.model == 'linear':
            # Handling for single data point scenario
            if len(heights) == 1:
                LR = self.config['models']['temperature']['default-coefficient']
                h0 = heights[0]
                t0 = temperatures[0]
                # Creating a linear model using the default coefficient and single point
                self.coeffs = np.array([LR, t0 - LR * h0])
            else:
                # Fit a linear model to the data points
                self.coeffs = np.polyfit(heights, temperatures, 1)
            self.fitted = True
        else:
            raise ValueError('Unknown temperature model')

    def predict(self, heights):
        """
        Predict temperatures based on the fitted model for the provided heights.

        Args:
            heights (list): Heights for which to predict temperatures.

        Returns:
            list: Predicted temperatures.

        Raises:
            ValueError: If the model type is unknown or unsupported.
        """
        if self.model == 'linear':
            # Making predictions based on the linear model coefficients
            return np.polyval(self.coeffs, heights)
        else:
            raise ValueError('Unknown temperature model')


class WindModel:
    """
    Represents a wind model, typically linear, that can be fitted to data points and used for predictions.

    Attributes:
        config (dict): Configuration dictionary containing model parameters.
        model (str): Type of model to be used, retrieved from the configuration.
        coeffs (list): Coefficients of the fitted model.
        fitter (bool): Flag indicating if the model has been fitted (not actively used in the current implementation).
    """

    def __init__(self, config):
        """Initialize the WindModel with the given configuration."""
        self.config = config
        self.model = config['models']['wind']['model']
        self.coeffs = []
        self.fitted = False

    def fit(self, heights, speeds):
        """
        Fit the model based on the provided heights and wind speeds.

        Args:
            heights (list): List of height measurements.
            speeds (list): List of corresponding wind speed measurements.

        Raises:
            ValueError: If heights and speeds don't have the same length, are empty, or if the model type is unknown.
        """
        if len(heights) != len(speeds):
            raise ValueError('Heights and wind speeds must have the same length')
        if len(heights) == 0:
            raise ValueError('Heights and wind speeds must have at least one element')

        if self.model == 'linear':
            # Handling for single data point scenario
            if len(heights) == 1:
                LR = self.config['models']['wind']['default-coefficient']
                h0 = heights[0]
                s0 = speeds[0]
                # Creating a linear model using the default coefficient and single point
                self.coeffs = np.array([LR, s0 - LR * h0])
            else:
                # Fit a linear model to the data points
                self.coeffs = np.polyfit(heights, speeds, 1)
            self.fitted = True
        else:
            raise ValueError('Unknown wind model')

    def predict(self, heights):
        """
        Predict wind speeds based on the fitted model for the provided heights.

        Args:
            heights (list): Heights for which to predict wind speeds.

        Returns:
            list: Predicted wind speeds.

        Raises:
            ValueError: If the model type is unknown or unsupported.
        """
        if self.model == 'linear':
            # Making predictions based on the linear model coefficients
            return np.polyval(self.coeffs, heights)
        else:
            raise ValueError('Unknown wind model')

class AirPressureModel():
    def __init__(self, config):
        self.config = config
        self.model = config['models']['air-pressure']['model']
        self.coeffs = []
        self.fitted = False

    def fit(self, temperature_model):
        if self.model == 'barometric':
            max_height = self.config['models']['numerical']['max-height']
            num = self.config['models']['numerical']['integration-precision']
            g = self.config['models']['constants']['gravitational-acceleration']
            R = self.config['models']['constants']['gas-constant']
            M = self.config['models']['constants']['molar-mass-air']
            P0 = self.config['models']['constants']['atm-pressure']

            heights = np.linspace(0, max_height, num=num)
            dh = max_height/num
            temps = (temperature_model.predict(heights) - 32)*5/9 + 273.15
            integrals = np.zeros(len(heights))
            for i, h in enumerate(heights[1:], 1):
                integrals[i] = integrals[i-1] + dh/temps[i]
            pressures = P0*np.exp(-g*M/R*integrals)*0.01
            self.interpolated_function = interp1d(heights, pressures)

            self.fitted = True
        else:
            raise ValueError('Unknown Air Pressure model')

    def predict(self, heights):
        if self.model == 'barometric':
            return self.interpolated_function(heights)
        else:
            raise ValueError('Unknown Air Pressure model')
    
class AirDensityeModel():
    def __init__(self, config):
        self.config = config
        self.model = config['models']['air-density']['model']
        self.coeffs = []
        self.fitted = False

    def fit(self, pressure_model, temperature_model):
        if self.model == 'barometric':
            R_d = self.config['models']['constant']['specific-gas-constant']
            
            self.interpolated_function = lambda x: pressure_model.predict(x)/(R_d*((temperature_model.predict(x) - 32)*5/9 + 273.15))

            self.fitted = True
        else:
            raise ValueError('Unknown Air Density model')

    def predict(self, heights):
        if self.model == 'barometric':
            return self.interpolated_function(heights)
        else:
            raise ValueError('Unknown Air Density model')
class CloudModel():
    def __init__(self, config):
        self.config = config
        self.fitted = False

    def fit(self, cloud_levels, clouds):
        def interpolate_cloud_value(height):
            # Filter out any NaN values from the arrays
            valid_indices = ~np.isnan(clouds) & ~np.isnan(cloud_levels)
            filtered_clouds = np.array(clouds)[valid_indices]
            filtered_levels = np.array(cloud_levels)[valid_indices]
            
            # Ensure the arrays are sorted by cloud level
            sorted_indices = np.argsort(filtered_levels)
            sorted_clouds = filtered_clouds[sorted_indices]
            sorted_levels = filtered_levels[sorted_indices]

            # If height is outside the range, clip it to the range
            height = np.clip(height, sorted_levels[0], sorted_levels[-1])

            # Interpolate the cloud value at the specified height
            interpolated_value = np.interp(height, sorted_levels, sorted_clouds)

            return interpolated_value
        self.interpolated_function = interpolate_cloud_value
        self.fitted = True
    def predict(self, heights):
        return self.interpolated_function(heights)