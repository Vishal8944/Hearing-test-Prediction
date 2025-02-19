import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
import pickle
from sklearn.preprocessing import MinMaxScaler




data=pd.read_csv("hearing_test.csv")
array = data.values
X = array[:, 0:-1]
scaler = MinMaxScaler()
scaler.fit(X)

loaded_model = load(open('Logmodel', 'rb'))



def HEARING_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    input_data_reshaped=scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 1):
      return '1 '
    else:
      return '0 '
    



def main():
    
    
    # giving a title
    st.title('Model Deployment: LOGISTIC REGRESSION Model')
    
    
    # getting the input data from the user
    
    
    number1 = st.number_input('Insert AGE', min_value=0, max_value=85, value=0, step=1, format="%d")
    number2 = st.number_input('Insert PHYSICAL SCORE', min_value=0, max_value=200, value=0, step=1, format="%d")
    


    TEST_RESULT = ''
    
    # creating a button for Prediction
    
    if st.button(' Test Result'):
        TEST_RESULT = HEARING_prediction([number1, number2])
        
        
    st.success(TEST_RESULT)
    
    
    
    
    
if __name__ == '__main__':
    main()