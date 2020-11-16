import pandas as pd
import os
import logging

from sklearn.preprocessing import StandardScaler
from pickle import dump

def preprocessing(data):
    
    #separate features and target as x & y
    y = data['target']
    x = data.drop('target', axis = 1)
    
    num_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    num_features = x[num_columns]

    #scale the values using a StandardScaler
    print('Applying Standard Scaler...')

    scaler = StandardScaler()
    scaler = scaler.fit(num_features)
    X = scaler.transform(num_features)

    #features DataFrame 
    features = pd.DataFrame(X, columns = num_columns)

    print('Finish preprocessing with success!')
    
    save_features(scaler, "StandardScaler")

    return features, y

def readData(): 
    data = pd.read_csv('trainingapp/data/raw/heart.csv')
    return data

def save_features(preprocessor, filename):

    basepath = os.path.join(os.path.abspath(""), "trainingapp/preprocess/")

    # Save pickle model
    out_file = os.path.join(basepath, filename + ".pkl")
    dump(preprocessor, open(out_file, "wb"))
    print(f"Model saved to: {out_file}")
    logging.debug(f"Model saved: {out_file}")
    return True

if __name__ == "__main__":
    data = readData()
    preprocessing(data)