import numpy as np
import numba
from metpy.units import units
from metpy.io import parse_metar_to_dataframe

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

def cloud_model(heights, cloud_types, cloud_levels, config):
    def cloud_type_to_numeric(condition):
        mapping = {
            'CLR': 0.0,
            'FEW': 0.2,
            'SCT': 0.4,
            'BKN': 0.6,
            'OVC': 0.8,
            'VV': 1.0
        }
        if condition not in mapping.keys():
            return 0.0
        return mapping[condition]
    
    h = heights

    cloud_types = np.array([cloud_type_to_numeric(x) for x in cloud_types])
    valid_indices = ~np.isnan(cloud_levels)

    filtered_clouds = cloud_types[valid_indices]
    filtered_levels = cloud_levels[valid_indices]

    if len(filtered_clouds) == 0:
        return 0.0
    
    sorted_indices = np.argsort(filtered_levels)
    sorted_clouds = filtered_clouds[sorted_indices]
    sorted_levels = filtered_levels[sorted_indices]

    # If height is outside the range, clip it to the range
    h = np.clip(h, sorted_levels[0], sorted_levels[-1])

    # Interpolate the cloud value at the specified height
    return np.interp(h, sorted_levels, sorted_clouds)

def severity_model(heights, wxcodes):
    bad_codes = ['FC', #funnel clouds (tornados)
                 'DS', #dust storm
                 'SS', #Sandstorm
                 'FZRA', #freezing rain
                 '+SN', #heavy snow
                 'GR', #hail
                 '+RA', #heavy rain
                 'TS', #thunderstorm,
                 'CG', #cloud-ground lightening
                 'IC', #in-cloud lightening
                 'LTG', #lightening,
                 '+FC', #funnel clouds (tornados)
                 '+DS', #dust storm
                 '+SS', #Sandstorm
                 '+FZRA', #freezing rain
                 '+GR', #hail
                 '+TS', #thunderstorm,
                 '+CG', #cloud-ground lightening
                 '+IC', #in-cloud lightening
                 '+LTG', #lightening
                 '-FC', #funnel clouds (tornados)
                 '-DS', #dust storm
                 '-SS', #Sandstorm
                 '-FZRA', #freezing rain
                 '-GR', #hail
                 '-TS', #thunderstorm,
                 '-CG', #cloud-ground lightening
                 '-IC', #in-cloud lightening
                 '-LTG' #lightening
                 ]
    
    for code in wxcodes:
        if code in bad_codes:
            return 1.
    return 0.

def estimate_scalars(heights, scalar, data_dict, config):
    metar_df = interpret_metar(data_dict['METAR'])
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
        # Cloud levels are in feet, so we need to convert them to meters.
        cloud_levels = metar_df[['low_cloud_level', 'medium_cloud_level', 'high_cloud_level', 'highest_cloud_level']].values.flatten()
        cloud_types = metar_df[['low_cloud_type', 'medium_cloud_type', 'high_cloud_type', 'highest_cloud_type']].values.flatten()
        return cloud_model(heights, cloud_types, cloud_levels/3.28084 + data_dict['elevation'], config)
    elif scalar == 'severity':
        return severity_model(heights, metar_df[['current_wx1', 'current_wx2', 'current_wx3']].values.flatten())
    else:
        raise ValueError('Unknown scalar')
    
def interpret_metar(metar_data):
    if not metar_data.startswith('METAR'):
        metar_data = 'METAR ' + metar_data
    return parse_metar_to_dataframe(metar_data)

