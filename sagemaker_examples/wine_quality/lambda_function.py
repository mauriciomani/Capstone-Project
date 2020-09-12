import os
import io
import boto3
import json
import csv

# I Create the variable called API_SAGEMAKER and as value de saagemaker endpoint
api_sagemaker = os.environ['API_SAGEMAKER']
#with te following line you can later invoke sagemaker endpoint
runtime= boto3.client('runtime.sagemaker')

#when you create a lamnda function you have to specify a lambda handler.And basically is executed when lambda invokes de code
def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    #make sure that received data is converted to json format
    data = json.loads(json.dumps(event))
    #extract what is inside data key
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=api_sagemaker,
                                       ContentType='text/csv',
                                       Body=payload)
    print('Initializing!')
    print(response)
    result = json.loads(response['Body'].read().decode())
    print('Wine quality is:')
    print(result)
    #since it will return an array, extract the first element
    pred = int(result[0])
    # If prediction is bigger than 5 then "wine is good" output: this was completely arbitrary.
    predicted_label = 'Wine is good' if pred > 5 else 'Bad quality wine' 
    
    return predicted_label