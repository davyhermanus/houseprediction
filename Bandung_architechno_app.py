import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Load dataset
column_names = [
    'Architecture', 'Location', 'Land_Area', 'Building_Area', 'Bedrooms',
    'Bathrooms', 'Living_Room', 'Kitchen_Quality', 'Security', 'Greenness',
    'Damage', 'House_Age', 'Flood_Risk', 'Price'
]
data = pd.read_csv('hargaprediksijualrumahnotitle.csv', header=None, delim_whitespace=True, names=column_names)

# Features and target
X = data[column_names[:-1]]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar - User input
st.sidebar.header("Input House Features")

def user_input_features():
    return pd.DataFrame({
        'Architecture': [st.sidebar.slider('Architecture', 100, 200, 150)],
        'Location': [st.sidebar.slider('Location (1-5)', 1, 5, 3)],
        'Land_Area': [st.sidebar.slider('Land Area (m²)', 100, 500, 250)],
        'Building_Area': [st.sidebar.slider('Building Area (m²)', 50, 300, 150)],
        'Bedrooms': [st.sidebar.slider('Bedrooms', 1, 6, 3)],
        'Bathrooms': [st.sidebar.slider('Bathrooms', 1, 5, 2)],
        'Living_Room': [st.sidebar.slider('Living Rooms', 1, 3, 1)],
        'Kitchen_Quality': [st.sidebar.slider('Kitchen Quality (1-3)', 1, 3, 2)],
        'Security': [st.sidebar.slider('Security (0-3)', 0, 3, 1)],
        'Greenness': [st.sidebar.slider('Greenness (0-3)', 0, 3, 2)],
        'Damage': [st.sidebar.slider('Damage Level (0-5)', 0, 5, 1)],
        'House_Age': [st.sidebar.slider('House Age (years)', 0, 100, 10)],
        'Flood_Risk': [st.sidebar.slider('Flood Risk (0-4)', 0, 4, 1)],
    })

input_df = user_input_features()

# Page title
st.title("🏡 Bandung House Price Prediction")
st.markdown("This app predicts **house prices in Bandung** based on key property features. Adjust the sliders to simulate different property conditions.")

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(input_df)
st.subheader("💰 Predicted House Price:")
st.success(f"Rp {prediction[0]:,.0f}")

# Feature Importance
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
st.pyplot(fig)

# Evaluation Metrics
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.subheader("📈 Model Performance on Test Data")
st.markdown(f"""
- **R-squared**: {r2:.2f}  
  → Proportion of variance in price explained by the model.
  
- **Mean Absolute Error (MAE)**: Rp {mae:,.0f}  
  → Average absolute difference between prediction and true price.

- **Root Mean Squared Error (RMSE)**: Rp {rmse:,.0f}  
  → Like MAE but penalizes large errors more.

- **Mean Absolute Percentage Error (MAPE)**: {mape:.2f}%  
  → Average percentage error. Closer to 0% is better.

- **Explained Variance Score**: {evs:.2f}  
  → How much of price variance is captured. Higher is better.
""")
