import os
import streamlit as st
import pandas as pd

from load.load_pipeline import load_sklearn_object
from PIL import Image

def launch():
    """
    Main function to launch all functionalities of Streamlit    
    """

    try:
        main_interface()
        side_menu()
        return True
    except:
        print('Error')
        return False
    
def main_interface():
    # Add Streamlit title, descriptions and load an image
    st.title('Heart Disease App')
    st.write('Insert your data and see the results. This is a sample application and cannot be used as a substitute for real mediacl advice.')

    BASEPATH = os.path.abspath("images")
    IMAGE = "heart_section.jpg"
    image_path = os.path.join(BASEPATH, IMAGE)
    image = Image.open(image_path)
    st.image(image, width=None)

    st.write('Please fill in the details of the the person under consideration in the left sidebar and click on the button below!')

def side_menu():
    st.sidebar.title('Insert your data')
    
    age = st.sidebar.number_input("age", 1, 110, 25, 1)
    trestbps = st.sidebar.slider("resting blood pressure", 50, 250, 130, 1)
    chol = st.sidebar.slider("serum cholestoral in mg/dl", 100, 600, 250, 1)
    talac = st.sidebar.slider("maximum heart rate achieved", 50, 250, 150, 1)
    oldpeak = st.sidebar.number_input("ST depression induced by exercise relative to rest", 0.0, 10.0, 1.0, 0.1)

    row = [age, trestbps, chol, talac, oldpeak]

    if (st.sidebar.button('Submit')):
        print('Button clicked!')
        feat_cols = ['age', 'trestbps', 'chol', 'talac', 'oldpeak']

        feature_trasformation = load_sklearn_object("StandardScaler.pkl", 'preprocess')
        model = load_sklearn_object("DecisionTreeClassifier-auto.pkl", 'models')
        
        # Create the Dataframe
        features = pd.DataFrame([row], columns = feat_cols)
        
        # Feature Engineering
        data = feature_trasformation.transform(features)
        print("feat eng ok")
        # Prediction
        predictions = model.predict(data)
        
        if(predictions[0] == 0):
            st.write("This is a healthy person!")
        else:
            st.write("This person has high chances of having diabetes!")
    