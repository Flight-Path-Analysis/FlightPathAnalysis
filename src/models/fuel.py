from traffic.core import Flight
import numpy as np
import datetime

def compute_fuel_data(state_vectors, icao24):
    """Compute fuel rates for each flight segment."""


    state_vectors['altitude'] = state_vectors['baroaltitude']
    state_vectors['TAS'] = state_vectors['velocity']

    # add vertical_rate
    state_vectors['vertical_rate'] = state_vectors['altitude'].diff().fillna(0)

    # state_vectors.rename(columns={'time': 'timestamp'}, inplace=True)

    state_vectors['timestamp'] = state_vectors['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    state_vectors['icao24'] = icao24

    state_vectors = Flight(state_vectors).fuelflow().data

    if 'fuel' not in state_vectors.columns:
        state_vectors['fuel'] = np.repeat(np.nan, len(state_vectors))
    if 'fuelflow' not in state_vectors.columns:
        state_vectors['fuelflow'] = np.repeat(np.nan, len(state_vectors))
    if 'mass' not in state_vectors.columns:
        state_vectors['mass'] = np.repeat(np.nan, len(state_vectors))
    if 'dt' not in state_vectors.columns:
        state_vectors['dt'] = np.repeat(np.nan, len(state_vectors))

    state_vectors['used_fuel'] = state_vectors['fuel'].copy()
    cols_to_drop = ['altitude', 'TAS', 'vertical_rate', 'timestamp', 'icao24', 'fuel', 'dt']
    for col in cols_to_drop:
        if col in state_vectors.columns:
            state_vectors.drop(col, axis=1, inplace=True)
     
    return state_vectors


