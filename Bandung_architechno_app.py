import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Bandung Real Estate Price Prediction App
This app predicts the **Bandung Real Estate Price**!
""")

# Load dataset
column_names = ['ARS', 'LKS', 'LT', 'LB', 'KT', 'KM', 'RT', 'DPR', 'KMN', 'ASR', 'RSK', 'UMR', 'BJR', 'HARGA']
feature_names = ['ARS', 'LKS', 'LT', 'LB', 'KT', 'KM', 'RT', 'DPR', 'KMN', 'ASR', 'RSK', 'UMR', 'BJR']
data = pd.read_csv(r'hargaprediksijualrumahnotitle.csv', header=None, delimiter=r"\s+", names=column_names)
st.write('Dataset Source')
st.write(data.head())

# Splitting features and target
X = data[feature_names]
Y = data['HARGA']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Sidebar for user input
st.sidebar.header('Input Parameters')

def user_input_features():
    ARS = st.sidebar.slider('ASR - Bentuk Arsitektur', float(X.ARS.min()), float(X.ARS.max()), float(X.ARS.mean()))
    LKS = st.sidebar.slider('LKS - Lokasi Potensial dari rumah', 1, 5, 1)
    LT = st.sidebar.slider('LT - Luas Tanah dalam m2', float(X.LT.min()), float(X.LT.max()), float(X.LT.mean()))
    LB = st.sidebar.slider('LB - Luas Bangunann dalam m2', float(X.LB.min()), float(X.LB.max()), float(X.LB.mean()))
    KT = st.sidebar.slider('KT - Jumlah kamar Tidur', 3, 5, 1)
    KM = st.sidebar.slider('KM - Jumlah kamar Mandi', 2, 4, 1)
    RT = st.sidebar.slider('RT - Jumlah ruang besar', 1, 3, 1)
    DPR = st.sidebar.slider('DPR - Tingkat kualitas Dapur', 1, 3, 1)
    KMN = st.sidebar.slider('KMN - Tingkat keamanan rumah', 0, 3, 1)
    ASR = st.sidebar.slider('ASR - Keasrian rumah', 0, 3)
    RSK = st.sidebar.slider('RSK - Tingkat kerusakan rumah', 0, 5, 1)
    UMR = st.sidebar.slider('UMR - Usia rumah (tahun)', 0, 10, 1)
    BJR = st.sidebar.slider('BJR - Kerawanan banjir', 0, 4, 1)
    
    data = {'ARS': ARS, 'LKS': LKS, 'LT': LT, 'LB': LB, 'KT': KT, 'KM': KM, 'RT': RT, 'DPR': DPR, 'KMN': KMN, 'ASR': ASR, 'RSK': RSK, 'UMR': UMR, 'BJR': BJR}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input features
st.write('User Input Parameters:')
st.write(df)

# Build the Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make prediction
prediction = model.predict(df)

# Display the prediction
st.write(f'Prediksi Harga Rumah: {prediction[0]:,.2f}')
