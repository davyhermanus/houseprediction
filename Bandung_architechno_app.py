import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Load data
column_names = [
    'Architecture', 'Location', 'Land_Area', 'Building_Area', 'Bedrooms',
    'Bathrooms', 'Living_Room', 'Kitchen_Quality', 'Security', 'Greenness',
    'Damage', 'House_Age', 'Flood_Risk', 'Price'
]
df = pd.read_csv("hargaprediksijualrumahnotitle.csv", header=None, delim_whitespace=True, names=column_names)

X = df.drop(columns=["Price"]).copy()
X['Security'] *= 0.3
y = df["Price"]

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# --- UI ---
st.title("🏠 Bandung House Price Prediction & Compatibility")

# Input Features
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

# Preference Input
st.sidebar.header("🎯 Your Preferences")
st.sidebar.markdown("Rate importance of each feature (1–5)")
user_pref = {
    key: st.sidebar.slider(f"{key} Importance", 1, 5, 3) for key in X.columns
}

# Scale user input
user_input_scaled = scaler.transform(user_input)
user_scaled_df = pd.DataFrame(user_input_scaled, columns=X.columns)

# === Compatibility Score (Standardized Method) ===
pref_series = pd.Series(user_pref)
pref_std = (pref_series - pref_series.mean()) / pref_series.std()
pref_std_norm = pref_std / pref_std.abs().sum()
raw_score = np.dot(user_scaled_df.values[0], pref_std_norm.values)
compatibility_score = 1 / (1 + np.exp(-raw_score))  # sigmoid

# === Prediction ===
predicted_price = model.predict(user_input_scaled)[0]

# --- Output ---
st.subheader("💰 Predicted Price")
st.success(f"Rp {predicted_price:,.0f}")

st.subheader("🧩 Compatibility Score")
st.info(f"{compatibility_score * 100:.1f}% match")

# --- Feature Importance ---
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
st.pyplot(fig)

# --- Model Evaluation ---
st.subheader("📈 Model Evaluation")
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f"""
- **R-squared (R²)**: `{r2:.2f}` – Model fit quality.
- **Mean Absolute Error (MAE)**: `Rp {mae:,.0f}` – Avg. error in prediction.
- **Root Mean Squared Error (RMSE)**: `Rp {rmse:,.0f}` – Penalizes large errors.
- **MAPE**: `{mape:.2f}%` – Avg. % error.
- **Explained Variance Score**: `{evs:.2f}` – Variation explained.
""")

# --- Final Explanation ---
st.subheader("📝 Final Recommendation")
mean_price = y.mean()
std_price = y.std()
if predicted_price > mean_price + std_price:
    price_level = "high"
elif predicted_price < mean_price - std_price:
    price_level = "low"
else:
    price_level = "average"

if compatibility_score > 0.8:
    match_level = "strong"
elif compatibility_score > 0.6:
    match_level = "medium"
else:
    match_level = "low"

if price_level == "high" and match_level == "low":
    comment = "⚠️ Expensive and doesn't align with your preferences."
elif price_level == "low" and match_level == "strong":
    comment = "🎉 Great match and affordable!"
else:
    comment = "📌 Consider this house based on your trade-offs."

st.info(comment)

# --- Done ---
