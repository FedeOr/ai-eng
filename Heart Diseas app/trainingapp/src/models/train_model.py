import logging
import os
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from pickle import dump


RANDOM_SEED = 42

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("Dataset splitted")
    print(f"X_train: {X_train.shape}")
    print(f"X_train: {X_test.shape}")
    print(f"X_train: {y_train.shape}")
    print(f"X_train: {y_test.shape}")

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    logging.debug(f"Dataset splitted: {result}")

    return result

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse

def save_model(model, filename):

    basepath = os.path.join(os.path.abspath(""), "trainingapp/models/")

    # Save pickle model
    out_file = os.path.join(basepath, filename + ".pkl")
    dump(model, open(out_file, "wb"))
    print(f"Model saved to: {out_file}")
    logging.debug(f"Model saved: {out_file}")
    return True

def train(split_dictionary, max_features):
    
    X_train = split_dictionary["X_train"]
    y_train = split_dictionary["y_train"]
    X_test = split_dictionary["X_test"]
    y_test = split_dictionary["y_test"]
    
    # Define model
    model = DecisionTreeClassifier(max_features=max_features)
    
    #Fit model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(eval_metrics(y_test, y_pred))
    
    # Save model
    model_name = "DecisionTreeClassifier-"+max_features
    save_model(model, model_name)
    
    
    
