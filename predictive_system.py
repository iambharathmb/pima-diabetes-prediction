import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# load the saved model
model = pickle.load(open("model.sav", "rb"))

# making a Prediction 
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# Prediction
prediction = model.predict(input_data_reshape)

if prediction[0]==0:
    print("The Person is not Diabetes")
else:
    print("The Person is Diabetes")