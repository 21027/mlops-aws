# in this file we will make the preprocessing 

import pandas as pd
import sys
import yaml
import os

## load params from yaml
params = yaml.safe_load(open("params.yaml"))['preprocess']



def preprocess(input_path,output_path):
    os.makedirs(os.path.dirname(output_path),exist_ok=True) # if does not exist


    data = pd.read_csv(input_path)
    #? any data preprocessing needed should be done here
    data = data.dropna()
    
    data.to_csv(output_path,index=False)
    print(f"Preprocessed data saved to :{output_path}")

if __name__== "__main__" :
    

    # input_path,output_path = params["input"],params["output"]

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    preprocess(input_path,output_path)
