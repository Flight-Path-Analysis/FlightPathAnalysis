# Flight Path Analysis

## Description

Flight Path Analysis is a comprehensive project developed as part of the Erdös Institute's Data Science Bootcamp. The project allows users to download data from the OpenSky database (provided the user has the required credentials) and aims to organize and compress this data based on a precision factor. Though still in its initial stages, the ultimate goal is to download weather data (the method is yet to be decided) and perform statistical analysis, correlating deviations in flight paths with weather conditions. Additionally, the project aspires to build and train a machine-learning tool to predict extra fuel costs due to weather deviations.

Currently, only the data-gathering component is available.

## Documentation

Detailed documentation for `Flight Path Analysis` can be found [here](https://flightpathanalysis.readthedocs.io/en/latest/).

## Features

- **Data Gathering**: Simplified packages for downloading data from OpenSky and potential weather data services.
- **Data Compression**: Flight Path and Weather Data compression, simplification, and encoding.
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

Feel free to check [Contribution Guidelines](LINK_TO_CONTRIBUTING.md) for more details (if available).

## License

This project is not currently licensed.

## Contact

For any queries or feedback, you can contact me through GitHub or drop an email at [andrerg01@gmail.com](mailto:andrerg01@gmail.com).

## Acknowledgments

- **Erdös Institute**: For hosting the Data Science Bootcamp where this project was conceptualized.
- **OpenSky**: For providing access to their comprehensive flight database.