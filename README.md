# Stanford_2021_Fall_Battery_Trajectory_Prediction_Project
This repository contains code for our work on early trajectory prediction of battery lifetime. Python is the main language used for all code below.
The data is not open_sourced yet, please contact jihan123@stanford.edu if you are interesed in learning more about our project
### Our key scripts and functions are summarized here:

## AWS_version_local
All codes are embedded in to a `battery_model` and you can just interact with the model with `final_note.ipynb` or run all the hyperparameter tuning process with `aws_version.py`. Some examples methods files includes:
  `B_dataprocessing.py`: Inlcude functions to Proprocess all 390 cells data and extract the best fitting curve X and y.

  `B_plotting.py`: Include three function to plot the Empirical Degredation curve, Empirical Selected Fitting Degredation curve, and Predicted Degredation curve, and correlation Heatmap between variables.  

  `B_DNN_model.py`: Include functions of Multi-output Neural Network that can be implement to train the battery data.

  `B_notebook.ipynb`: It includes the step by step implementation of the project.
  
  `Final.ipynb`: implement class directly, used for hyperparameter tuning

## AWS_Version_results
downlaod from the aws vm after all results were generated

