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
data = pd.read_csv("hargaprediksijualrumahnotitle.csv", header=None, delim_whitespace=True, names=column_names)

# Preprocess
X = data.drop(columns=["Price"]).copy()
X['Security'] *= 0.3  # Normalize security
y = data["Price"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Get historical means (for initial values)
feature_means = X.mean()

# UI
st.title("🏠 Bandung House Price Prediction & Compatibility App")
st.markdown("This app predicts house prices and evaluates how well they fit your preferences based on historical data.")

# User Input
st.sidebar.header("🏡 House Features")
user_input = pd.DataFrame([{
    'Architecture': st.sidebar.slider('Architecture Score', 100, 200, int(feature_means['Architecture'])),
    'Location': st.sidebar.slider('Location (1–5)', 1, 5, int(round(feature_means['Location']))),
    'Land_Area': st.sidebar.slider('Land Area (m²)', 100, 600, int(feature_means['Land_Area'])),
    'Building_Area': st.sidebar.slider('Building Area (m²)', 50, 400, int(feature_means['Building_Area'])),
    'Bedrooms': st.sidebar.slider('Bedrooms', 1, 6, int(round(feature_means['Bedrooms']))),
    'Bathrooms': st.sidebar.slider('Bathrooms', 1, 5, int(round(feature_means['Bathrooms']))),
    'Living_Room': st.sidebar.slider('Living Rooms', 1, 3, int(round(feature_means['Living_Room']))),
    'Kitchen_Quality': st.sidebar.slider('Kitchen Quality (1–3)', 1, 3, int(round(feature_means['Kitchen_Quality']))),
    'Security': st.sidebar.slider('Security Level (0–3)', 0, 3, int(round(feature_means['Security'] / 0.3))) * 0.3,
    'Greenness': st.sidebar.slider('Greenness (0–3)', 0, 3, int(round(feature_means['Greenness']))),
    'Damage': st.sidebar.slider('Damage (0–5)', 0, 5, int(round(feature_means['Damage']))),
    'House_Age': st.sidebar.slider('House Age', 0, 100, int(round(feature_means['House_Age']))),
    'Flood_Risk': st.sidebar.slider('Flood Risk (0–4)', 0, 4, int(round(feature_means['Flood_Risk']))),
}])

# Preferences
st.sidebar.header("🎯 Your Preferences")
st.sidebar.markdown("Set importance for each feature (1 = low, 5 = high)")
user_pref = {key: st.sidebar.slider(f"{key} Importance", 1, 5, 3) for key in X.columns}
pref_series = pd.Series(user_pref)
pref_std = (pref_series - pref_series.mean()) / pref_series.std()
pref_norm = pref_std / pref_std.abs().sum()

# Prediction
user_scaled = scaler.transform(user_input)
predicted_price = model.predict(user_scaled)[0]

# Compatibility
raw_score = np.dot(user_scaled[0], pref_norm)
compatibility_score = 1 / (1 + np.exp(-raw_score))

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

# Model Evaluation
st.subheader("📈 Model Evaluation Metrics")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f"""
- **R² Score**: `{r2:.2f}` — Variance explained by the model  
- **MAE**: `Rp {mae:,.0f}` — Mean Absolute Error  
- **MSE**: `Rp {mse:,.0f}` — Mean Squared Error  
- **RMSE**: `Rp {rmse:,.0f}` — Root of MSE  
- **MAPE**: `{mape:.2f}%` — Average percent error  
- **Explained Variance**: `{evs:.2f}` — Similar to R² but less sensitive to outliers
""")

# Final Narrative
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

msg = "📌 Standard option"
if price_level == "high" and match_level == "strong":
    msg = "💰 High price, but excellent match!"
elif price_level == "low" and match_level == "strong":
    msg = "🎉 Great match and affordable!"
elif match_level == "low":
    msg = "⚠️ May not align with your preferences."

st.info(msg)

# Matched/mismatched features
matched, mismatched = [], []
for feature in X.columns:
    user_val = user_input[feature].values[0]
    mean_val = feature_means[feature]
    pref = pref_series[feature]
    if pref >= 4 and user_val < mean_val:
        mismatched.append(feature)
    elif pref <= 2 and user_val > mean_val:
        mismatched.append(feature)
    else:
        matched.append(feature)

if matched:
    st.markdown(f"**✅ Matched Preferences:** {', '.join(matched)}")
if mismatched:
    st.markdown(f"**❌ Potential Mismatches:** {', '.join(mismatched)}")

# Ranked Houses
X_all_scaled = scaler.transform(X)
all_scores = 1 / (1 + np.exp(-np.dot(X_all_scaled, pref_norm)))
data['Compatibility (%)'] = (all_scores * 100).round(2)
top_houses = data.sort_values(by='Compatibility (%)', ascending=False)

st.subheader("🏘️ Top 3 Recommended Houses")
for i, (_, row) in enumerate(top_houses.head(3).iterrows(), 1):
    st.markdown(f"#### 🏡 House #{i}")
    st.markdown(f"- **Price**: `Rp {row['Price']:,.0f}`")
    st.markdown(f"- **Match**: `{row['Compatibility (%)']}%`")
    st.markdown(f"- **Bedrooms**: {row['Bedrooms']} | Bathrooms: {row['Bathrooms']}")
    st.markdown(f"- **Land**: {row['Land_Area']} m² | Building: {row['Building_Area']} m²")
