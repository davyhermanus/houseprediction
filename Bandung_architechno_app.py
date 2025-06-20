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

# Modify Security weight
X = data.drop(columns=["Price"]).copy()
X['Security'] = X['Security'] * 0.3  # Reduce influence of Security
y = data['Price']

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("🏠 House Parameters")
def get_user_input():
    user_data = {
        'Architecture': st.sidebar.slider('Architecture (score)', 100, 200, 150),
        'Location': st.sidebar.slider('Location (1–5)', 1, 5, 3),
        'Land_Area': st.sidebar.slider('Land Area (m²)', 100, 500, 250),
        'Building_Area': st.sidebar.slider('Building Area (m²)', 50, 300, 150),
        'Bedrooms': st.sidebar.slider('Bedrooms', 1, 6, 3),
        'Bathrooms': st.sidebar.slider('Bathrooms', 1, 5, 2),
        'Living_Room': st.sidebar.slider('Living Rooms', 1, 3, 1),
        'Kitchen_Quality': st.sidebar.slider('Kitchen Quality (1–3)', 1, 3, 2),
        'Security': st.sidebar.slider('Security Level (0–3)', 0, 3, 1) * 0.3,  # apply reduction here too
        'Greenness': st.sidebar.slider('Greenness (0–3)', 0, 3, 2),
        'Damage': st.sidebar.slider('Damage Level (0–5)', 0, 5, 1),
        'House_Age': st.sidebar.slider('House Age (years)', 0, 100, 10),
        'Flood_Risk': st.sidebar.slider('Flood Risk (0–4)', 0, 4, 1),
    }
    return pd.DataFrame(user_data, index=[0])

# Get input
user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)

# Predict
prediction = model.predict(user_input_scaled)

# App title
st.title("🏡 Bandung House Price Prediction")
st.markdown("This app predicts house prices in **Bandung** based on key property features. Adjust the sliders to simulate property conditions.")

# Show prediction
st.subheader("💰 Predicted House Price")
st.success(f"Rp {prediction[0]:,.0f}")

# Feature importance
st.subheader("📊 Feature Importance (After Reducing Security Influence)")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="pastel", ax=ax)
st.pyplot(fig)

# Evaluate model
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

# Show metrics with explanation
st.subheader("📈 Model Performance on Test Data")

st.markdown(f"""
### R-squared (R²)
**R²: `{r2:.2f}`**  
R² measures how well the model explains the variance in the actual prices.  
A value closer to **1** indicates that the model is performing well.

---

### Mean Absolute Error (MAE)
**MAE: `Rp {mae:,.0f}`**  
This is the average of the absolute differences between predicted and actual prices.  
A lower MAE means more accurate predictions.

---

### Root Mean Squared Error (RMSE)
**RMSE: `Rp {rmse:,.0f}`**  
This metric gives more weight to large errors (because it squares them).  
It tells you roughly how far off your predictions are, in Rupiah.

---

### Mean Absolute Percentage Error (MAPE)
**MAPE: `{mape:.2f}%`**  
This shows the average percentage error between predictions and actual prices.  
For example, a MAPE of 10% means predictions are off by 10% on average.

---

### Explained Variance Score
**Explained Variance: `{evs:.2f}`**  
Indicates how much of the price variance is captured by the model.  
A value close to 1 is preferred and means the model fits the data well.
""")
