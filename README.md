## Summary
A project written in Python to see how several sklearn machine learning algorithms handle loan data to evaluate data features and predict default.

## Details
This project is intended to provide some introductory insights into machine learning applications within finance by leveraging several common and
readily approachable machine learning algorithms from the sklearn library to evaluate loan statistics.
The application employs the following algorithms:  
  - Extra Trees Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Gaussian Naive Bayes Classifier

The input data set should be in csv format; it must have a column header called 'loan status' which should represent your target variable, and a
column called 'member_id'.  All other data features are potential dependent variables, and there are no limits or requirements around those features.
For ease of implementation, the application expects all input vales to be floats or integers.

The application will divide the input data into training and testing subsets, and generate three to five output files in .csv format, based on user selections:
  - A prediction file which will include the all the dependent variable values from the test set, the actual result (target variable) and a column of predictions
    for each algorithm
  - A feature evaluation file with the ranked feature importance scores for the ensemble algorithms (Extra Trees, and Gradient Boosting), which include this property
  - A recursive feature elimnation file, which displays the rank order of eliminated features for each algorithm, with the exception of Naive Bayes, which does not
    surface the necessary feature attributes to enable RFE
  - If selected by the user, the application will eliminate any dependent variables which were determined to have 0.0 predictive contirbution by the Extra Trees
    and Gradient Boosting feature evaluation process, and will re-run each of the predictions again, using the refined list of dependent variables creating two new output files with the source of the refined dependent variable features list specified in the file name.

The application will also allow users to review prediction results on screen, presenting the number of incorrectly predicted data points, a classification report, and
confusion matrices with and without normalization.  The confusion matrices will also be plotted for some light visualization.  

If you elect not to review results on screen, you will not see the classification or confusion matrices, but prediciton results, feature evaluation scores and
recursive feature elimination ranks are included in the output files.  

## Prerequisites
This package assumes the user is using Python 2.x.  There are likely some adjustments necessary to the base code to enable Python 3.

Expected package dependencies are listed in the "requirements.txt" file for PIP.  User will need to run the following:

```
pip install -r requirements.txt
```

As configured, the application requires a directory structure as established in this repository.  The code will not run if output directories matching the filepaths in
the code are not present.  However, filepaths are configured as global variables in the first portion of the code and can be specified there.  

## Installation
Download the source code:

```shell
git clone https://github.com/rjzap/freestyle
cd freestyle\pd_predict_master
```

Ensure the example data upload file is available in the data directory, otherwise place a desired file in the data folder and update the input_filepath variable in the
source code.  

## Usage
To run the app, from the pd_predict_master working directory, in your shell:

```
python app\pd_predict_app.py
```
