import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Load dataset
column_names = [
    'Architecture', 'Location', 'Land_Area', 'Building_Area', 'Bedrooms',
    'Bathrooms', 'Living_Room', 'Kitchen_Quality', 'Security', 'Greenness',
    'Damage', 'House_Age', 'Flood_Risk', 'Price'
]
data = pd.read_csv("hargaprediksijualrumahnotitle.csv", header=None, delim_whitespace=True, names=column_names)

# Preprocess
X = data.drop(columns=["Price"]).copy()
X["Security"] *= 0.3
y = data["Price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# App UI
st.title("🏠 Bandung House Price Prediction & Compatibility App")

st.sidebar.header("🏡 House Features")
user_input = pd.DataFrame([{
    'Architecture': st.sidebar.slider('Architecture Score', 100, 200, 150),
    'Location': st.sidebar.slider('Location (1–5)', 1, 5, 3),
    'Land_Area': st.sidebar.slider('Land Area (m²)', 100, 500, 250),
    'Building_Area': st.sidebar.slider('Building Area (m²)', 50, 300, 150),
    'Bedrooms': st.sidebar.slider('Bedrooms', 1, 6, 3),
    'Bathrooms': st.sidebar.slider('Bathrooms', 1, 5, 2),
    'Living_Room': st.sidebar.slider('Living Rooms', 1, 3, 1),
    'Kitchen_Quality': st.sidebar.slider('Kitchen Quality (1–3)', 1, 3, 2),
    'Security': st.sidebar.slider('Security Level (0–3)', 0, 3, 1) * 0.3,
    'Greenness': st.sidebar.slider('Greenness (0–3)', 0, 3, 2),
    'Damage': st.sidebar.slider('Damage (0–5)', 0, 5, 1),
    'House_Age': st.sidebar.slider('House Age', 0, 100, 10),
    'Flood_Risk': st.sidebar.slider('Flood Risk (0–4)', 0, 4, 1),
}])

st.sidebar.header("🎯 Your Preferences")
st.sidebar.markdown("Adjust importance (1–5) for each feature.")
user_pref = {
    key: st.sidebar.slider(f"{key} Importance", 1, 5, 3)
    for key in X.columns
}
# Safe fallback if all preferences are same
if len(set(user_pref.values())) == 1:
    pref_series = pd.Series({k: 1 for k in X.columns})
else:
    pref_series = pd.Series(user_pref)
    pref_series = (pref_series - pref_series.mean()) / pref_series.std()
pref_norm = pref_series / pref_series.abs().sum()

# Prediction
user_scaled = scaler.transform(user_input)
predicted_price = model.predict(user_scaled)[0]

# Compatibility Score
dot_score = np.dot(user_scaled[0], pref_norm)
compatibility_score = 1 / (1 + np.exp(-dot_score))

# Display Prediction
st.subheader("💰 Predicted Price")
st.success(f"Rp {predicted_price:,.0f}")

st.subheader("🧩 Compatibility Score")
st.info(f"{compatibility_score * 100:.1f}% match")
