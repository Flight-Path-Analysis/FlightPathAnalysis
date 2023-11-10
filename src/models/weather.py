import numpy as np
from scipy.interpolate import interp1d
from src.common import utils
import numba

@numba.njit
def temperature_model(heights, elevation, temperature, LR):
    # Lapse Rate Model
    h = heights
    T0 = temperature
    h0 = elevation
    return T0 + LR * (heights - h0)

@numba.njit
def wind_speed_model(heights, elevation, wind_speed, G):
    # Gradient Wind Model
    h = heights
    T0 = wind_speed
    h0 = elevation
    return T0 + G * (heights - h0)

@numba.njit
def air_pressure_model(heights, elevation, temperature, LR, g, R, M, P0):
    T0 = temperature
    T0 = (T0 - 32) * 5/9 + 273.15
    h0 = elevation
    h = heights
    integral = 1/LR*(np.log(1 - (h*LR)/(h0*LR - T0)))
    return P0*np.exp(-g*M/R*integral)*0.01
    
@numba.njit
def air_density_model(heights, elevation, temperature, LR, g, R, M, P0, Rd):
    P = air_pressure_model(heights, elevation, temperature, LR, g, R, M, P0)*100
    T = temperature_model(heights, elevation, temperature, LR)
    T = (T - 32) * 5/9 + 273.15
    return P/(Rd*T)

def cloud_model(heights, cloud_coverages, cloud_levels, config):
    if isinstance(heights, list) or isinstance(heights, np.ndarray):
        return np.array([cloud_model(x, cloud_coverages, cloud_levels, config) for x in heights])
    def sky_condition_to_numeric(condition):
        mapping = {
            'CLR': 0.0,
            'FEW': 0.2,
            'SCT': 0.4,
            'BKN': 0.6,
            'OVC': 0.8,
            'VV': 1.0
        }
        # Return the numeric value corresponding to the condition, NaN if not found
        return mapping.get(condition, 0.0)
    
    h = heights

    cloud_coverages = np.array([sky_condition_to_numeric(x) for x in cloud_coverages])
    valid_indices = ~np.isnan(cloud_coverages) & ~np.isnan(cloud_levels)

    filtered_clouds = np.array(cloud_coverages)[valid_indices]
    filtered_levels = np.array(cloud_levels)[valid_indices]

    if len(filtered_clouds) == 0:
        return 0.0
    
    sorted_indices = np.argsort(filtered_levels)
    sorted_clouds = filtered_clouds[sorted_indices]
    sorted_levels = filtered_levels[sorted_indices]

    # If height is outside the range, clip it to the range
    h = np.clip(h, sorted_levels[0], sorted_levels[-1])

    # Interpolate the cloud value at the specified height
    interpolated_value = np.interp(h, sorted_levels, sorted_clouds)

    return interpolated_value

def severity_model(wxcode):
    # Handle missing or nan values
    if not wxcode or wxcode == 'nan':
        return 0

    # Base score for different weather phenomena
    base_scores = {
        '+': 0.6,  # Heavy intensity
        '-': 0.2,  # Light intensity
        'RA': 0.3,  # Rain
        'SN': 0.5,  # Snow
        'TS': 0.7,  # Thunderstorm
        'FZ': 0.6,  # Freezing
        'DZ': 0.1,  # Drizzle
        'BR': 0.2,  # Mist
        'FG': 0.4,  # Fog
        'UP': 0.3,  # Unknown Precipitation
        'HZ': 0.2,  # Haze
        'FU': 0.1,  # Smoke
        'VCTS': 0.5,  # Vicinity Thunderstorm
        'BCFG': 0.3,  # Patches of Fog
        'MIFG': 0.3,  # Shallow Fog
    }

    score = 0.0
    components = wxcode.split()

    # Calculate the score based on weather phenomena and their intensity
    for comp in components:
        for key, value in base_scores.items():
            if key in comp:
                score += value
                break  # Break to avoid double counting (e.g., 'RA' in '+RA')

    # Normalize the score to be between 0 and 1
    score = min(score, 1.0)
    return score


def estimate_scalars(heights, scalar, data_dict, config):
    if scalar == 'tmpf':
        LR = config['models']['temperature']['default-coefficient']
        return temperature_model(heights, data_dict['elevation'], data_dict['tmpf'], LR)
    elif scalar == 'sknt':
        G = config['models']['wind']['default-coefficient']
        return wind_speed_model(heights, data_dict['elevation'], data_dict['sknt'], G)
    elif scalar == 'air_pressure':
        LR = config['models']['temperature']['default-coefficient']
        g = config['models']['constants']['gravitational-acceleration']
        R = config['models']['constants']['gas-constant']
        M = config['models']['constants']['molar-mass-air']
        P0 = config['models']['constants']['atm-pressure']
        return air_pressure_model(heights, data_dict['elevation'], data_dict['tmpf'], LR, g, R, M, P0)
    elif scalar == 'air_density':  
        LR = config['models']['temperature']['default-coefficient']
        g = config['models']['constants']['gravitational-acceleration']
        R = config['models']['constants']['gas-constant']
        M = config['models']['constants']['molar-mass-air']
        P0 = config['models']['constants']['atm-pressure']
        Rd = config['models']['constants']['specific-gas-constant']
        return air_density_model(heights, data_dict['elevation'], data_dict['tmpf'], LR, g, R, M, P0, Rd)
    elif scalar == 'clouds':
        return cloud_model(heights, data_dict['clouds'], data_dict['cloud_levels'], config)
    else:
        raise ValueError('Unknown scalar')
