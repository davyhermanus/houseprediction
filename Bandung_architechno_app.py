import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Modify Security
X = data.drop(columns=["Price"]).copy()
X['Security'] = X['Security'] * 0.3  # Reduce influence
y = data['Price']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Sidebar: User input
st.sidebar.header("🏡 House Features")
def get_user_input():
    features = {
        'Architecture': st.sidebar.slider('Architecture (score)', 100, 200, 150),
        'Location': st.sidebar.slider('Location (1–5)', 1, 5, 3),
        'Land_Area': st.sidebar.slider('Land Area (m²)', 100, 500, 250),
        'Building_Area': st.sidebar.slider('Building Area (m²)', 50, 300, 150),
        'Bedrooms': st.sidebar.slider('Bedrooms', 1, 6, 3),
        'Bathrooms': st.sidebar.slider('Bathrooms', 1, 5, 2),
        'Living_Room': st.sidebar.slider('Living Rooms', 1, 3, 1),
        'Kitchen_Quality': st.sidebar.slider('Kitchen Quality (1–3)', 1, 3, 2),
        'Security': st.sidebar.slider('Security Level (0–3)', 0, 3, 1) * 0.3,
        'Greenness': st.sidebar.slider('Greenness (0–3)', 0, 3, 2),
        'Damage': st.sidebar.slider('Damage Level (0–5)', 0, 5, 1),
        'House_Age': st.sidebar.slider('House Age (years)', 0, 100, 10),
        'Flood_Risk': st.sidebar.slider('Flood Risk (0–4)', 0, 4, 1),
    }
    return pd.DataFrame([features])

user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)

# Sidebar: Preferences
st.sidebar.header("🎯 Your Feature Preferences (Importance)")
st.sidebar.markdown("Rate how important each feature is to you (1 = not important, 5 = very important)")

preference_weights = {
    'Architecture': st.sidebar.slider('Architecture Importance', 1, 5, 3),
    'Location': st.sidebar.slider('Location Importance', 1, 5, 5),
    'Land_Area': st.sidebar.slider('Land Area Importance', 1, 5, 4),
    'Building_Area': st.sidebar.slider('Building Area Importance', 1, 5, 4),
    'Bedrooms': st.sidebar.slider('Bedrooms Importance', 1, 5, 3),
    'Bathrooms': st.sidebar.slider('Bathrooms Importance', 1, 5, 3),
    'Living_Room': st.sidebar.slider('Living Room Importance', 1, 5, 2),
    'Kitchen_Quality': st.sidebar.slider('Kitchen Quality Importance', 1, 5, 3),
    'Security': st.sidebar.slider('Security Importance', 1, 5, 5),
    'Greenness': st.sidebar.slider('Greenness Importance', 1, 5, 3),
    'Damage': st.sidebar.slider('Damage Importance', 1, 5, 2),
    'House_Age': st.sidebar.slider('House Age Importance', 1, 5, 2),
    'Flood_Risk': st.sidebar.slider('Flood Risk Importance', 1, 5, 3),
}

# Normalize preferences
pref_series = pd.Series(preference_weights)
pref_norm = pref_series / pref_series.sum()

# Calculate compatibility score
compat_score = np.dot(user_input.values[0], pref_norm.values) / 5
compat_score = min(1.0, max(0.0, compat_score))

# Prediction
predicted_price = model.predict(user_input_scaled)[0]

# App UI
st.title("🏠 Bandung House Price Prediction & User Match")
st.write("This app predicts house prices and also shows how well a house matches your personal preferences.")

st.subheader("💰 Predicted Price")
st.success(f"Rp {predicted_price:,.0f}")

st.subheader("🧩 Compatibility Score with Your Preferences")
st.info(f"{compat_score * 100:.1f}% match")

# Feature importance plot
st.subheader("📊 Feature Importance in Price Prediction")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
st.pyplot(fig)

# Model Evaluation
st.subheader("📈 Model Evaluation on Test Data")

y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f"""
**R-squared (R²)**: `{r2:.2f}`  
Measures how well the model explains the variability in price. Closer to 1 is better.

**Mean Absolute Error (MAE)**: `Rp {mae:,.0f}`  
Average absolute difference between predicted and actual prices. Lower is better.

**Root Mean Squared Error (RMSE)**: `Rp {rmse:,.0f}`  
Similar to MAE but penalizes large errors more heavily. In Rupiah.

**Mean Absolute Percentage Error (MAPE)**: `{mape:.2f}%`  
Average percentage error. Below 20% is generally considered good.

**Explained Variance Score**: `{evs:.2f}`  
Indicates how much variance is captured by the model. Closer to 1 is better.
""")
