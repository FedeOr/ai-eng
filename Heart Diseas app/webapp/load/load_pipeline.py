import os
from pickle import load
import pandas as pd
import streamlit as st


def load_sklearn_object(object_name, folder):
    # Import sklearn pickle objects from training pipeline (model and feature eng pipelines)
    folder_path = os.path.abspath(".")
    training_path = os.path.abspath("../../trainingapp")

    sklearn_object_path = os.path.join(training_path, folder, object_name)

    print(f"sklearn object name read: {sklearn_object_path}")

    object_result = pd.read_pickle(sklearn_object_path)

    return object_result