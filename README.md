# Stanford_2021_Fall_Battery_Trajectory_Prediction_Project
This repository contains code for our work on early trajectory prediction of battery lifetime. Python is the main language used for all code below.

Our key scripts and functions are summarized here:

B_dataprocessing.py: Inlcude functions to Proprocess all 390 cells data and extract the best fitting curve X and y.
B_plotting.py: Include three function to plot the Empirical Degredation curve, Empirical Selected Fitting Degredation curve, and Predicted Degredation curve, and correlation Heatmap between variables.
B_DNN_model.py: uses the AdaBoostRegressor module from scikit-learn. We performed a grid-search over the n_trees and learning_rate hyperparameters to select the optimal values. For each number of cycles we use 5-fold cross-validation to calculate the mean percent error and mean standard error. Saves the trained models so they can be used for testing.
Notebook_Implementation.ipynb: It includes the step by step implementation of the project.
