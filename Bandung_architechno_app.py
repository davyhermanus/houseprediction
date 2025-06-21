
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

# Feature Importance
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
st.pyplot(fig)

# Metrics
st.subheader("📈 Model Metrics")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)
st.markdown(f'''
- **R² Score**: `{r2:.2f}` — Variance explained by model.
- **MAE**: `Rp {mae:,.0f}` — Mean absolute error.
- **RMSE**: `Rp {rmse:,.0f}` — Root of squared error.
- **MAPE**: `{mape:.2f}%` — Mean percent error.
- **Explained Variance**: `{evs:.2f}`
''')

# Final narrative
st.subheader("📝 Final Assessment")
price_level = "average"
if predicted_price > y.mean() + y.std():
    price_level = "high"
elif predicted_price < y.mean() - y.std():
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

# --- Feature Importance ---
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
st.pyplot(fig)

# --- Model Evaluation Metrics ---
st.subheader("📈 Model Evaluation")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f'''
- **R² Score**: `{r2:.2f}` — Proportion of variance explained.
- **MAE**: `Rp {mae:,.0f}` — Average absolute error.
- **MSE**: `Rp {mse:,.0f}` — Mean of squared errors.
- **RMSE**: `Rp {rmse:,.0f}` — Root of MSE.
- **MAPE**: `{mape:.2f}%` — Error in percent.
- **Explained Variance Score**: `{evs:.2f}`
''')

# --- Final Assessment ---
st.subheader("📝 Final Assessment")
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

matched = []
mismatched = []
for feature in X.columns:
    val = user_input[feature].values[0]
    pref = user_pref[feature]
    mean_val = X[feature].mean()
    if pref >= 4 and val >= mean_val:
        matched.append(feature)
    elif pref <= 2 and val < mean_val:
        matched.append(feature)
    else:
        mismatched.append(feature)

if price_level == "high" and match_level == "strong":
    message = "💰 High price, but excellent match!"
elif price_level == "low" and match_level == "strong":
    message = "🎉 Great match and affordable!"
elif match_level == "low":
    message = "⚠️ May not align with your preferences."
else:
    message = "📌 Reasonably priced, worth a look."

st.info(message)
if matched:
    st.markdown(f"**✅ Matched Features:** `{', '.join(matched)}`")
if mismatched:
    st.markdown(f"**❌ Mismatched Features:** `{', '.join(mismatched)}`")

# --- Compatibility for all houses ---
X_all_scaled = scaler.transform(X)
dot_products = np.dot(X_all_scaled, pref_norm.values)
compat_scores = 1 / (1 + np.exp(-dot_products))
data['Compatibility (%)'] = (compat_scores * 100).round(2)
ranked_data = data.sort_values(by='Compatibility (%)', ascending=False)

st.subheader("🏘️ Top House Recommendations")
for i in range(3):
    house = ranked_data.iloc[i]
    st.markdown(f"### 🏡 House #{i+1}")
    st.markdown(f"- **Price**: `Rp {house['Price']:,.0f}`")
    st.markdown(f"- **Compatibility**: `{house['Compatibility (%)']}%`")
    st.markdown(f"- **Bedrooms**: {house['Bedrooms']} | **Bathrooms**: {house['Bathrooms']}")
    st.markdown(f"- **Location Score**: {house['Location']} | **Security**: {house['Security'] / 0.3:.0f}")
    st.markdown(f"- **Size**: Land {house['Land_Area']} m², Building {house['Building_Area']} m²")
