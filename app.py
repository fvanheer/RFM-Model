
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import plotly.express as px

DATA_FILE = 'rfm_segments.csv'

st.title("Interactive Plot to Analysis Final RFM Segments")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_FILE)
    data = data[(data['MonetaryValue'] > 0) & (data['Recency'] <=360) & (data['Frequency'] <= 100)]
    return data

data = load_data()
recency = st.sidebar.number_input('Smaller Than Recency', 0, 360, 180)
frequency= st.sidebar.number_input('Smaller Than Frequency', 0, 100, 25)
monetaryValue = st.sidebar.number_input('Smaller Than Monetary Value', 0, 100000, 25000)

data = data[(data['Recency']<=recency) & (data['Frequency']<=frequency) & (data['MonetaryValue']<=monetaryValue)]

data

#scatter plot
fig_scatter = px.scatter(data, x="Recency", y="Frequency", color="Customer Segment",
                 size='MonetaryValue', hover_data=['R_Quartile', 'F_Quartile', 'M_Quartile'])

st.plotly_chart(fig_scatter)

#show distribution of values
#recency
fig_r = px.histogram(data, x="Recency", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=data.columns, title='Recency Plot')
st.plotly_chart(fig_r)

#frequency
fig_f = px.histogram(data, x="Frequency", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=data.columns, title='Frequency Plot')
st.plotly_chart(fig_f)

#monetary value
fig_m = px.histogram(data, x="MonetaryValue", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=data.columns, title='Monetary Value Plot')
st.plotly_chart(fig_m)
