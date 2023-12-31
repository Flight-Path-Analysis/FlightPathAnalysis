# Flight Path Analysis

## Description

Flight Path Analysis is a comprehensive project developed as part of the Erdös Institute's Data Science Bootcamp. The project allows users to download data from the OpenSky database (provided the user has the required credentials) and the Iowa Environmental Mesonet and aims to organize and compress this data based on a precision factor. Then, this repository can be used to perform statistical analysis (correlate deviations in flight paths with weather conditions), as well as to build and train a machine-learning tool to predict extra fuel costs due to weather deviations.


## Documentation

Detailed documentation for `Flight Path Analysis` can be found [here](https://flightpathanalysis.readthedocs.io/en/latest/).

## Features

- **Data Gathering**: Simplified packages for downloading data from OpenSky and potential weather data services.
- **Data Compression**: Flight Path and Weather Data compression, simplification, and encoding using relational databases/sqlite.
- **Statistical Analysis**: Analyze the correlation between flight paths and weather conditions.
- **Machine Learning**: A workflow to train and test a predictive machine learning algorithm for assessing fuel costs.

## Installation

1. Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

2. For users with OpenSky database access:
    - Create a `.yml` file in the `config` directory.
    - Update `config.yml` with the appropriate paths and credentials.
      Here's a snippet for reference:
      ```yaml
      base-configs:
          root-directory: '/path_to_your_directory/FlightPathAnalysis/'
          tag: 'test'
          opensky-credentials: 'path_to_opensky_credentials.yml'
          weather-credentials: 'path_to_weather_credentials.yml'
      ```

3. For users without OpenSky access:
    - Download example flight data from [this link (will be available soon)](here.link.will.exist.flight).
    - Download example weather data from [this link (will be available soon)](here.link.will.exist.weather).

## Usage

*Information on usage will be available soon.*

## Contributing

We welcome contributions to the Flight Path Analysis project! Here are some general guidelines:

- Fork the repository and create your branch from `master`.
- Ensure the tests pass on your branch.
- Make sure your code lints (use tools like `flake8` or `pylint`).
- Issue that pull request!

Feel free to check [Contribution Guidelines](https://github.com/Andrerg01/FlightPathAnalysis/blob/main/docs/CONTRIBUTING.md) for more details (if available).

The files pertaining to the [Code of Conduct](https://github.com/Andrerg01/FlightPathAnalysis/blob/main/docs/CODE_OF_CONDUCT.md), [License](https://github.com/Andrerg01/FlightPathAnalysis/blob/main/docs/LICENSE), [Contribution Guidelines](https://github.com/Andrerg01/FlightPathAnalysis/blob/main/docs/CONTRIBUTING.md), and [Security](https://github.com/Andrerg01/FlightPathAnalysis/blob/main/docs/SECURITY.md) can all be found inside the repo's [docs](https://github.com/Andrerg01/FlightPathAnalysis/tree/main/docs) directory.

## License

Flight Path Analysis © 2023 by Andre Guimaraes is licensed under CC BY-NC-SA 4.0.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/

## Contact

For any queries or feedback, you can contact a team member through GitHub or drop an email at [andrerg01@gmail.com](mailto:andrerg01@gmail.com).

## Acknowledgments

- **Erdös Institute**: For hosting the Data Science Bootcamp where this project was conceptualized.
- **OpenSky**: For providing access to their comprehensive flight database.
