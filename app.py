import joblib
import numpy as np
import pandas as pd
import streamlit as st

# load model
model = joblib.load('ridge.joblib')

# extract list of address
df = pd.read_csv('data/housePrice.csv')
address = sorted(df.Address.dropna().unique())

# create form to get data from user
st.title('Tehran House Price Prediction Form')
area = st.number_input('Enter Area', value=70, step=1, min_value=0, max_value=1000)
room_count = st.number_input('Enter Room', value=1, step=1, min_value=0, max_value=6)
location = st.selectbox("Choose Address", address)
parking = st.checkbox('Parking')
warehouse = st.checkbox('Warehouse')
elevator = st.checkbox('Elevator')

# define predict method to prediction price
def predict(): 
    columns = df.columns.drop(['Address', 'Price', 'Price(USD)']).to_list() + address
    row = np.array([area, room_count, int(parking), int(warehouse), int(elevator)])
    row = np.concatenate((row, (np.array(address) == location).astype(int))).tolist()
    
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)

    st.info("{:,.2f} Toman".format(prediction[0]))

st.button('Predict Price', on_click=predict)