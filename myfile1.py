import pickle
import streamlit as st

model1 = pickle.load(open("areaprice.pkl","rb"))

def mydeploy():
    st.title("Area Price Prediction")
    area=st.number_input("Enter Area in square feet:")
    pred=st.button("Predict Price")

    if pred:
        X=model1.predict([[area]])
        st.write("price of given area is:", X[0])
        
mydeploy()
