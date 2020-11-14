import os
import logging

def debug_path():
    # Configure basepath and define dataset path (debug only)
    BASEPATH = os.path.abspath("trainingapp/data/raw")
    DATASET = "heart.csv"

    dataset_path = os.path.join(BASEPATH, DATASET)
    logging.debug(f"Dataset path: {dataset_path}")

    return dataset_path