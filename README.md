# Ecce Homo
Ecce Homo aims to be an automatic tool to implement machine learning workflow tasks, often call as AutoML. To see an example please visit [eccehomo](https://github.com/mauriciomani/Capstone-Project/tree/master/eccehomo) folder where you will find a IPython Notebbok with implementation on titanic dataset.
This is part of the Final Capstone project on the Machine learning engineer nanodegree. For more information please visit [Udacity](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).
Or visit either [proposal.pdf](https://github.com/mauriciomani/Capstone-Project/blob/master/proposal.pdf) or [project_report.pdf](https://github.com/mauriciomani/Capstone-Project/blob/master/project_report.pdf).
## Why Ecce Homo?
If you want to easily establish a benchmark for any particular model you are working on, try Ecce Homo, to fastly implement a machine learning model. Also, if you are a newby in machine learning and want to create your first product give it a try.
## What can Ecce Homo do?
It can perform automatic exploratory analysis tasks and export them to a html file such as:
* Dataset description, number of rows and columns, name of the features. 
* Summary statistics, mean, median, first and third quartile, minimum and maximum values, as well as count. 
* Group by, aggregation on categorical variables, for fast decisions and understanding of data classes. 
* Boxplot plot categorical variables to have a deep understanding of the distribution of numerical data with information on the target class as well. 
* Histograms plot the distribution of continuous data to understand distribution and compare distributions between the classes of target variables. 
* Give brief understanding of empty values on each variable. 
* Correlations print a correlation matrix with all numerical variables plus a heatmap to visually find highly correlated features. 
* Show pairwise scatter plots on n random features (to easily visualize all features on high dimensional spaces). To look for patterns in data. 
* Bar plots on categorical data to understand how classes are distributed among. 


You can also perform automatic data imputation with just four lines of code for train and test separetly to keep test integrity. Calling DataImputation class<br>
`imputation = eccehomo.DataImputation(X_train, X_test)`<br>
Defining imputation for train set:<br>
`X_train = imputation.user_defined(defined_methods = {"Embarked":"mode", "Cabin":"indicator", "Age":"knn"}, indicator = 'other')`<br>
Performing imputation on test set, either by using same method or same value as in train:<br>
`X_test = imputation.test_imputation(imputation.imputed_values, use_value = True)`<br>
Finally print information of imputed values:<br>
`imputation.imputed_values`<br>

You can perform outlier detection using isolation forest, DBSCAN, discretization or boxplot method. With one line of code:<br>
`X = eccehomo.Outlier(X_train).isolation_forest(max_outliers = 1000)`<br>

Also perform **Bayesian Optimization** to search more efficently on a predifined hyperparamters space. Using one line of code:<br>
`model = eccehomo.Modeling(X_train, y_train, algo = "RandomForest", iter = 50, scoring='precision').optimize()`<br>
For more information on Bayesian optimization, please visit: https://github.com/fmfn/BayesianOptimization.

## What you need:
In order to perform the examples found on the examples folder. You need the following libraries and datsets.
* Sklearn version 0.22.1
* pandas 0.25.1
* imblearn 0.6.1
* matplotlib 3.1.3
* seaborn 0.9.0
* numpy 1.17.2
* Installation of bayesian optimization through:<br>
`$ pip install bayesian-optimization`<br>

Datasets can be found in eccehomo/name/data folder or in the following Kaggle links to competitons:
* [Titanic](https://www.kaggle.com/c/titanic)
* [Churn](https://www.kaggle.com/c/ic20182)
* [Fraud](https://www.kaggle.com/c/competetion1/data)

## License
This project is built under the MIT license, for more information visit [here](https://github.com/mauriciomani/Capstone-Project/blob/master/LICENSE.txt).

## What is missing?
* Unit tests to workflow
* Improve data imputation and model hyperparameter tunning.
* Work on Balancing dataset module
* Model Evaluation
* Unit tests to repo and library.
