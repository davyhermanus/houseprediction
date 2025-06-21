import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import json
import os

# Load dataset
column_names = [
    'Architecture', 'Location', 'Land_Area', 'Building_Area', 'Bedrooms',
    'Bathrooms', 'Living_Room', 'Kitchen_Quality', 'Security', 'Greenness',
    'Damage', 'House_Age', 'Flood_Risk', 'Price'
]
data = pd.read_csv("hargaprediksijualrumahnotitle.csv", header=None, delim_whitespace=True, names=column_names)

# Prepare features and target
X = data.drop(columns=["Price"]).copy()
X['Security'] *= 0.3
y = data["Price"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# UI title
st.title("🏠 Bandung House Price Prediction & Compatibility App")

# Load default preferences
default_preferences = {}
if os.path.exists("default_preferences.json"):
    with open("default_preferences.json", "r") as f:
        default_preferences = json.load(f)

# Sidebar inputs
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

# User preferences
st.sidebar.header("🎯 Your Preferences")
user_pref = {
    key: st.sidebar.slider(
        f"{key} Importance", 1, 5,
        int(round(default_preferences.get(key, 3)))
    )
    for key in X.columns
}
pref_series = pd.Series(user_pref)
pref_std = (pref_series - pref_series.mean()) / pref_series.std()
pref_norm = pref_std / pref_std.abs().sum()

# Prediction
user_scaled = scaler.transform(user_input)
predicted_price = model.predict(user_scaled)[0]

# Compatibility score
raw_score = np.dot(user_scaled[0], pref_norm)
compatibility_score = 1 / (1 + np.exp(-raw_score))

st.subheader("💰 Predicted Price")
st.success(f"Rp {predicted_price:,.0f}")

st.subheader("🧩 Compatibility Score")
st.info(f"{compatibility_score * 100:.1f}% match")

# Feature importance
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
st.pyplot(fig)

# Evaluation metrics
st.subheader("📈 Model Evaluation")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f'''
- **R² Score**: `{r2:.2f}` — Explains variance.
- **MAE**: `Rp {mae:,.0f}` — Avg absolute error.
- **MSE**: `Rp {mse:,.0f}` — Squared error.
- **RMSE**: `Rp {rmse:,.0f}` — Root of MSE.
- **MAPE**: `{mape:.2f}%` — Percentage error.
- **Explained Variance Score**: `{evs:.2f}`
''')

# Final narrative
st.subheader("📝 Final Assessment")
mean_price = y.mean()
std_price = y.std()
price_level = "average"
if predicted_price > mean_price + std_price:
    price_level = "high"
elif predicted_price < mean_price - std_price:
    price_level = "low"

match_level = "low"
if compatibility_score > 0.8:
    match_level = "strong"
elif compatibility_score > 0.6:
    match_level = "medium"

if price_level == "high" and match_level == "strong":
    msg = "💰 High price, but excellent match!"
elif price_level == "low" and match_level == "strong":
    msg = "🎉 Great match and affordable!"
elif match_level == "low":
    msg = "⚠️ May not align with your preferences."
else:
    msg = "📌 Reasonably priced, worth a look."

st.info(msg)

# Compatibility per house
X_all_scaled = scaler.transform(X)
all_scores = 1 / (1 + np.exp(-np.dot(X_all_scaled, pref_norm)))
data['Compatibility (%)'] = (all_scores * 100).round(2)
df_sorted = data.sort_values(by='Compatibility (%)', ascending=False)

st.subheader("🏘️ Recommended Houses")
for i in range(3):
    house = df_sorted.iloc[i]
    st.markdown(f"#### 🏡 House #{i + 1}")
    st.markdown(f"- **Price**: `Rp {house['Price']:,.0f}`")
    st.markdown(f"- **Compatibility**: `{house['Compatibility (%)']}%`")
    st.markdown(f"- **Bedrooms**: {house['Bedrooms']} | **Bathrooms**: {house['Bathrooms']}")
    st.markdown(f"- **Location Score**: {house['Location']} | **Security**: {house['Security'] / 0.3}")
    st.markdown(f"- **Size**: Land {house['Land_Area']} m², Building {house['Building_Area']} m²")
