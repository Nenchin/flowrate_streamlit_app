# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.write("""
         # Flow Detector App
         """)
         
st.write("---")

model = joblib.load("lin_reg_model.pkl")

dp_1 = pd.DataFrame(np.load("sampled_data.npy"),
                    columns=["Pressure_In", "Flow_In",
                             "Pressure_Out", "Flow_Out"])

X = dp_1.drop("Flow_Out", axis=1)
y = dp_1["Flow_Out"]

st.sidebar.header("Input Parameters")

def input_features():
    Pressure_In = st.sidebar.slider("Pressure_In", X["Pressure_In"].min(),
                                     X["Pressure_In"].max(),
                                     X["Pressure_In"].mean())
    Flow_In = st.sidebar.slider("Flow_In", X["Flow_In"].min(),
                                X["Flow_In"].max(), X["Flow_In"].mean())
    Pressure_Out = st.sidebar.slider("Pressure_Out", X["Pressure_Out"].min(),
                                     X["Pressure_Out"].max(),
                                     X["Pressure_Out"].mean())
    data = {
            "Pressure_In": Pressure_In,
            "Flow_In": Flow_In,
            "Pressure_Out": Pressure_Out
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = input_features()

st.header("Specified Input Parameters")
st.write(df)
st.write("---")

prediction = model.predict(df)

st.header("Predicted outlet flow rate in l/m")
st.write(prediction)
st.write("---")