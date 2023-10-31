import numpy as np
# import utils

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
                                  If available, 'metar_tmpf', 'metar_elevation', 'metar_sknt_E', and 
                                  'metar_sknt_N' are also used.

    config (dict): A dictionary containing the configuration settings for the temperature and wind models, 
                   including the type of model and any necessary default coefficients.

    Returns:
    pd.DataFrame: The original stations_data DataFrame is returned, with additional columns for the models and 
                  sea-level predictions. New columns will be 'temperature_model', 'wind_model_E', 'wind_model_N', 
                  'tmpf_sea_level', 'sknt_E_sea_level', and 'sknt_N_sea_level'.
    """
    # Prepare a container for the models. Each station will have its own model.
    temperature_models = [TemperatureModel(config) for _ in range(len(stations_data))]
    wind_models_E = [WindModel(config) for _ in range(len(stations_data))]
    wind_models_N = [WindModel(config) for _ in range(len(stations_data))]

    # Iterate over stations data and fit models based on the data for each station.
    for i, row in stations_data.iterrows():
        elevations = [row['elevation']]
        temperatures = [row['tmpf']]
        wind_speeds_E = [row['sknt_E']]
        wind_speeds_N = [row['sknt_N']]
        # Adding metar data if available.
        if 'metar_tmpf' in row:
            elevations.append(row['metar_tmpf_elev'])
            temperatures.append(row['metar_tmpf'])
        if 'metar_sknt_E' in row and 'metar_sknt_N' in row:
            wind_speeds_E.append(row['metar_sknt_E'])
            wind_speeds_N.append(row['metar_sknt_N'])

        # Fit the model for this station.
        temperature_models[i].fit(elevations, temperatures)
        wind_models_E[i].fit(elevations, wind_speeds_E)
        wind_models_N[i].fit(elevations, wind_speeds_N)
    
    # Assign models to a new column in the DataFrame.
    stations_data['tmpf_model'] = temperature_models
    stations_data['sknt_E_model'] = wind_models_E
    stations_data['sknt_N_model'] = wind_models_N
    
    # Predict the sea-level temperature for each station using its respective model.
    stations_data['tmpf_sea_level'] = [row['tmpf_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['sknt_E_sea_level'] = [row['sknt_E_model'].predict([0])[0] for _, row in stations_data.iterrows()]
    stations_data['sknt_N_sea_level'] = [row['sknt_E_model'].predict([0])[0] for _, row in stations_data.iterrows()]

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
        self.fitted = False  # This attribute is set but not currently used

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
