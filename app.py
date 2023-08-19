import numpy as np
import pickle
import streamlit as st

# load the saved model
model = pickle.load(open("model.sav", "rb"))

# creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    # Prediction
    prediction = model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0] == 0):
        return 'The Person is not Diabetes'
    else:
        return 'The Person is Diabetes'
    
def main():
    # giving a title
    st.title("PIMA Diabetes Prediction Web App")
    
    # getting the input data from users
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressere Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    bmi = st.text_input("BMI value")
    DiabetesPedigreeFunction =st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the person")
    
    # code for prediction
    diagnosis = ''
    
    # craeting button for prediction
    if st.button('Result'):
        diagnosis = diabetes_prediction([
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            bmi,
            DiabetesPedigreeFunction,
            Age
        ])
        
    st.success(diagnosis)
            
if __name__ == '__main__':
    main()
    
    