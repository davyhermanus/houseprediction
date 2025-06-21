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

X = data.drop(columns=["Price"]).copy()
X['Security'] *= 0.3
y = data["Price"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("🏠 Bandung House Price Prediction & Preference Match")

# --- Sidebar Inputs ---
st.sidebar.header("🏡 House Features")

user_input = pd.DataFrame([{
    'Architecture': st.sidebar.slider('Architecture Score', 100, 200, 150),
    'Location': st.sidebar.slider('Location (1–5)', 1, 5, 3),
    'Land_Area': st.sidebar.slider('Land Area (m²)', 100, 500, 250),
    'Building_Area': st.sidebar.slider('Building Area (m²)', 50, 300, 150),
    'Bedrooms': st.sidebar.slider('Bedrooms', 1, 6, 2),
    'Bathrooms': st.sidebar.slider('Bathrooms', 1, 5, 1),
    'Living_Room': st.sidebar.slider('Living Rooms', 1, 3, 3),
    'Kitchen_Quality': st.sidebar.slider('Kitchen Quality (1–3)', 1, 3, 4),
    'Security': st.sidebar.slider('Security Level (0–3)', 0, 3, 3) * 0.3,
    'Greenness': st.sidebar.slider('Greenness (0–3)', 0, 3, 3),
    'Damage': st.sidebar.slider('Damage (0–5)', 0, 5, 3),
    'House_Age': st.sidebar.slider('House Age', 0, 100, 10),
    'Flood_Risk': st.sidebar.slider('Flood Risk (0–4)', 0, 4, 3),
}])

# --- User Preferences ---
st.sidebar.header("🎯 Your Preferences")
st.sidebar.markdown("Set the importance of each feature (1 = Not Important, 5 = Very Important)")
user_pref = {
    'Architecture': st.sidebar.slider('Architecture Importance', 1, 5, 3),
    'Location': st.sidebar.slider('Location Importance', 1, 5, 3),
    'Land_Area': st.sidebar.slider('Land Area Importance', 1, 5, 3),
    'Building_Area': st.sidebar.slider('Building Area Importance', 1, 5, 3),
    'Bedrooms': st.sidebar.slider('Bedrooms Importance', 1, 5, 2),
    'Bathrooms': st.sidebar.slider('Bathrooms Importance', 1, 5, 1),
    'Living_Room': st.sidebar.slider('Living Room Importance', 1, 5, 3),
    'Kitchen_Quality': st.sidebar.slider('Kitchen Quality Importance', 1, 5, 4),
    'Security': st.sidebar.slider('Security Importance', 1, 5, 3),
    'Greenness': st.sidebar.slider('Greenness Importance', 1, 5, 3),
    'Damage': st.sidebar.slider('Damage Importance', 1, 5, 3),
    'House_Age': st.sidebar.slider('House Age Importance', 1, 5, 5),
    'Flood_Risk': st.sidebar.slider('Flood Risk Importance', 1, 5, 3),
}
pref_series = pd.Series(user_pref)
pref_std = (pref_series - pref_series.mean()) / pref_series.std()
pref_norm = pref_std / pref_std.abs().sum()

# --- Prediction ---
user_scaled = scaler.transform(user_input)
predicted_price = model.predict(user_scaled)[0]

# --- Compatibility Score ---
raw_score = np.dot(user_scaled[0], pref_norm)
compatibility_score = 1 / (1 + np.exp(-raw_score))

st.subheader("💰 Predicted Price")
st.success(f"Rp {predicted_price:,.0f}")

st.subheader("🧩 Compatibility Score")
st.info(f"{compatibility_score * 100:.1f}% match")

# --- Feature Match/Mismatch ---
st.subheader("🔍 Feature Match Analysis")

matched = []
mismatched = []
for feature in X.columns:
    val = user_input[feature].values[0]
    avg = X[feature].mean()
    pref = pref_series[feature]
    if pref >= 4 and val >= avg:
        matched.append(feature)
    elif pref <= 2 and val <= avg:
        matched.append(feature)
    else:
        mismatched.append(feature)

if matched:
    st.markdown(f"✅ **Matches your preferences:** `{', '.join(matched)}`")
if mismatched:
    st.markdown(f"❌ **May not meet your preferences:** `{', '.join(mismatched)}`")

# --- Feature Importance ---
st.subheader("📊 Feature Importance in Price Prediction")
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
st.pyplot(fig)

# --- Model Metrics ---
st.subheader("📈 Model Performance Metrics")

y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f"""
- **R-squared (R²)**: `{r2:.2f}` – Measures how well the model explains price variation. Closer to 1 is better.
- **MAE (Mean Absolute Error)**: `Rp {mae:,.0f}` – Average absolute difference between predicted and actual prices.
- **MSE (Mean Squared Error)**: `Rp {mse:,.0f}` – Larger errors are penalized more.
- **RMSE (Root MSE)**: `Rp {rmse:,.0f}` – Interpretable in the same unit as price.
- **MAPE (Mean Absolute Percentage Error)**: `{mape:.2f}%` – Average prediction error in percentage.
- **Explained Variance Score**: `{evs:.2f}` – Higher is better, closer to 1.
""")
