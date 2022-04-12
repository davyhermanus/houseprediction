import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Bandung Real Estate Price Prediction App

This app predicts the ** Bandung Real Estate Price**!
""")
st.write('---')

# Loads Bandung House Price Dataset
column_names = ['ARS', 'LKS', 'LT', 'LB', 'KT', 'KM', 'RT', 'DPR', 'KMN', 'ASR', 'RSK', 'UMR', 'BJR', 'HARGA']
feature_names = ['ARS', 'LKS', 'LT', 'LB', 'KT', 'KM', 'RT', 'DPR', 'KMN', 'ASR', 'RSK', 'UMR', 'BJR']
data=pd.read_csv(r'hargaprediksijualrumahnotitle.csv',header=None, delimiter=r"\s+", names=column_names)
st.write('Tabel Sumber Dataset')
st.write(data.head())
X = pd.DataFrame(data, columns=feature_names)
st.write('Fitur Masukan')
st.write(X.head())
Y = pd.DataFrame(data, columns=["HARGA"])
st.write('Fitur Keluaran')
st.write(Y.head())

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Parameter Masukan yang dipilih')

def user_input_features():
    ARS = st.sidebar.slider('ASR - Bentuk Arsitektur',float(X.ARS.min()), float(X.ARS.max()),float( X.ARS.mean()))
    LKS = st.sidebar.slider('LKS - Lokasi Potensial dari rumah.', 1,5,1)
    LT = st.sidebar.slider('LT - Luas Tanah dalam m2', float(X.LT.min()), float(X.LT.max()),float( X.LT.mean()))
    LB = st.sidebar.slider('LB - Luas Bangunann dalam m2',float(X.LB.min()), float(X.LB.max()),float( X.LB.mean()))
    KT = st.sidebar.slider('KT - Jumlah kamar Tidur ', 3,5,1)
    KM = st.sidebar.slider('KM - Jumlah kamar Mandi', 2,4,1)
    RT = st.sidebar.slider('RT - Jumlah ruang besar ,ruang tamu, ruang keluarga ', 1,3,1)
    DPR = st.sidebar.slider('DPR - Tingkat kualitas Dapur dan Peralatannya',1,3,1)
    KMN = st.sidebar.slider('KMN - Tingkat keamanan perumahan dan jalan sekitar rumah', 0,3,1)
    ASR = st.sidebar.slider('ASR- Tingkat Keasrian dan adanya taman dari rumah', 0,3)
    RSK = st.sidebar.slider('RSK-  Tingkat kerusakan yang ada pada rumah', 0,5,1)
    UMR = st.sidebar.slider('UMR-  Usia rumah itu berdiri dalam tahunan', 0,10,1)
    BJR = st.sidebar.slider('BJR - Tingkat kerawanan Banjir dan Dampaknya',0,4,1)
    data = {'ARS-': ARS,
            'LKS': LKS,
            'LT': LT,
            'LB': LB,
            'KT': KT,
            'KM': KM,
            'RT': RT,
            'DPR': DPR,
            'KMN': KMN,
            'ASR': ASR,
            'RSK': RSK,
            'UMR': UMR,
            'BJR': BJR}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Hasil Pemilihan Input dari User')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.write('Prediksi dari  Harga dari Rumah Rata-rata')
st.write(prediction)
st.write('---')
