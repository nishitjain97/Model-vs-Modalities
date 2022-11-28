# Model vs Modalities
SBU CSE 512 Final Project to compare model performance for different data modalities.

## Directory Structure
* Raw Data: This is data that directly came from outside source

* Processed Data: This is the data that we have created from raw data and includes the ADS (etf_data.csv and gdp_data.csv) and preprocessed series for use in modeling (etf_data_diff_1.csv and gdp_data_diff_2.csv)

  * Note: For differentiated data, diff_1 implies first order differentiation and diff_2 implies second order differentiation to make data stationary. For diff_1, kindly ignore the first row before using it as the first row contains the original data that will be used when we want to integrate this to get the original data back. For diff_2, ignore the first two rows.
  
* Code: This contains code files and iPython notebooks.

## Useful Tips

* Code/utilities.py contains utility functions that can be used across the codebase. This includes:

  * data_integration() function will integrate differentiated data. For diff_1 data, pass the data through this function once to remove differencing. For diff_2 data, passing through function once will remove second differencing and passing the updated data again through the function will remove first differencing.
