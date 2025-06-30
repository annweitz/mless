# Homework 02

The second homework assignment deal with time series forecasting. This content of this folder are as follows:
* Notebook 1 to 6: contain the models, we were given, adjusted to use a shorter time-window because of RAM issues in
Google Colab
* Notebook 7: The multi-model plotter from exercise 1
* Notebook 8: Loading and preparing ozone and temperature data for exercise 2 and 3
* Notebook 9: The multivariate MLP model for exercise 2
* Notebook 10: The MLP to perform multivariate time series forecasting with known future covariates
* Results folder: contains the data used for the plots in Notebook 7

# Exercise 1
_Extend the code in notebook 7_multi_model_plotting to read the forecast results from all trained models and plot all 
curves in one panel. Make sure that your code can be re-used if you make forecasts for other stations or time episodes._

This exercise was mostly, pretty straight-forward. I reran the models from 3-6 to generate forecasts-results, albeit 
with a reduced time-window of 10 years (1997-2007) to counteract out of RAM issues, that arose with Google Colab. Running
the PatchTST model worked fine, until the last cells, that are supposed to reload the data samples for plot comparability. 
I couldn't resolve this issue in time, so this model is missing from the plots.  
The results of this exercise are displayed in Notebook 7. 

# Exercise 2
_Extend at least one of the models to input multivariate data. Download ozone data from TOAR (use the scripts in notebook 1). 
Copy the notebook with the model of your choice into a new notebook and extend the code so that temperazure and ozone 
data are used as inputs. The goal is to forecast ozone concentrations, you don't need to output temperature._

For this exercise, I choose to extend the MLP from notebook 4. At first, I downloaded ozone and temperature data to be 
used as input. The data was prepared with the preprocessing steps from the first notebook. This is done in Notebook 8. 
Additionally, I used drop_duplicates["timestamp"] after loading the normalized data back in in Notebook 9, since the 
dataframe contained duplicate timestamps, which resulted in the create_sequences function outputting empty data matrices. 
The issue being, that the function is looking for continuous windows, but the duplicates prevent this from working. 
For the multivariate data, both input variables are stacked. Since we only want ozone as output, we drop this later on. 
The model is a pretty simple MLP, analogous to the MLP used in notebook 4, but with a few more layers. 

## Exercise 3
_In reality, you will often have forecasts of weather variables (here temperature) available, so you can use future 
temperature values to forecast the ozone concentrations. Make another copy of your multivariate notebook and adjust 
the code accordingly. Use your multi-model plotter to compare the results from tasks 2 and 3._

For the last exercise, we are supposed to extend the multivariate model from the previous exercise, but also use data 
from future temperature values. This is also known as multivariate time series forecasting with known future covariates.
First we need to prepare the data. This is only a minor adjustment, because the create_sequences function, that we adjusted
for exercise 2 already returns everything we want for this. Additionally, we just need to split the future temperature 
values from the full variable target set, that we also use to get the ground truths. 

The model architecture gets a bit more complicated. Instead of building a Sequential model, I use the functional API 
to build a more complex one. It has two input layers, one for the past multivariate values, and one for the future 
covariate values. The past input gets passed through two Dense layers, before being concatenated with the flattened 
future input. After that, the combined vector is passed through another two dense layers, before being passed to an 
output layer to predict the future window.




