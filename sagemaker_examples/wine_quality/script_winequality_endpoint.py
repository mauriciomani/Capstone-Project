from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import subprocess
import sys
import numpy as np
from sklearn.linear_model import Lasso
from io import StringIO
from sagemaker_containers.beta.framework import (content_types, encoders, env, modules, transformer, worker)

#this might not be necessary since we have commented default bucket path
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
install('s3fs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--alpha', type=float, default = 0.8)
    parser.add_argument('--fit_intercept', type=bool, default = True)
    parser.add_argument('--max_iter', type=int, default = 1000)
    parser.add_argument('--random_state', type=int, default = 12)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    #https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()

    #Make sure to add the name of the fileto train since SM_CHANNEL_TRAIN looks inside input/data/train
    data = pd.read_csv(args.train + "/train.csv", header=None, engine="python") 

    # labels are in the first column.Leep aws logic. 
    #feel free to change input style
    y_train = data.iloc[:, 0]
    X_train = data.iloc[:, 1:]

    alpha = args.alpha
    random_state = args.random_state
    fit_intercept = args.fit_intercept
    max_iter = args.max_iter

    # Now use scikit-learn's lasso to train the model.
    lasso = Lasso(alpha = alpha, random_state = random_state, fit_intercept = fit_intercept, max_iter = max_iter)
    lasso = lasso.fit(X_train, y_train)

    # save the coefficients
    joblib.dump(lasso, os.path.join(args.model_dir, "model.joblib"))

#following functions are for prediction. The first (model_fn) makes sure to load the model
#The latter (input_fn) handles the data that is being inputed. In this case convert input to pandas dataframe
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    lasso = joblib.load(os.path.join(model_dir, "model.joblib"))
    return(lasso)

def input_fn(input_data, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header = None)
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))