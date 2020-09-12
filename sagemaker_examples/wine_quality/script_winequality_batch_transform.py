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
#However if plan to specify train default argument use this.
#We have commented those lines
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
    
    #parser.add_argument('--output-data-dir', type=str, default='s3://sagemaker-us-east-1-563718358426/tryouts/output')
    #parser.add_argument('--train', type=str, default='s3://sagemaker-us-east-1-563718358426/tryouts/input/data/training/train.csv')

    args = parser.parse_args()

    #Make sure to add the name of the path
    data = pd.read_csv(args.train + "/train.csv", header=None, engine="python") 

    # labels are in the first column
    y_train = data.iloc[:, 0]
    X_train = data.iloc[:, 1:]

    alpha = args.alpha
    random_state = args.random_state
    fit_intercept = args.fit_intercept
    max_iter = args.max_iter

    # Now use scikit-learn's decision tree classifier to train the model.
    lasso = Lasso(alpha = alpha, random_state = random_state, fit_intercept = fit_intercept, max_iter = max_iter)
    lasso = lasso.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(lasso, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

#Expect text_csv and transform to pandas
def input_fn(input_data, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header = None)
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))

#return all data plus prediction. We will use the model from model_fn and call predict.
def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    #pred_prob = model.predict_proba(input_data)
    input_data["prediction"] = np.array(prediction)
    #input_data["predict_proba"] = np.array(pred_prob)
    return(input_data)
    #return np.array(prediction)

#Use pandas to_csv to export as csv to output path in s3
def output_fn(prediction, content_type):
    #prediction.dumps()
    return(prediction.to_csv(index=False))
    #return worker.Response(encoders.encode(prediction, content_type), mimetype=content_type)