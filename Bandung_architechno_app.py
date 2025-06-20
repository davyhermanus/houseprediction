import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Set the page title and layout
st.set_page_config(page_title="Bandung Real Estate Price Prediction", layout="wide")

# App title and description
st.write("""
# Bandung Real Estate Price Prediction
This app predicts the **Bandung Real Estate Price** based on several features. Simply adjust the sliders on the left to see the predicted price of a house.
""")

# Load dataset
column_names = ['Architecture', 'Location', 'Land_Area', 'Building_Area', 'Bedrooms', 'Bathrooms', 'Living_Room', 'Kitchen_Quality', 'Security', 'Greenness', 'Damage', 'House_Age', 'Flood_Risk', 'Price']
feature_names = ['Architecture', 'Location', 'Land_Area', 'Building_Area', 'Bedrooms', 'Bathrooms', 'Living_Room', 'Kitchen_Quality', 'Security', 'Greenness', 'Damage', 'House_Age', 'Flood_Risk']
data = pd.read_csv('hargaprediksijualrumahnotitle.csv', header=None, delimiter=r"\s+", names=column_names)
st.write('Dataset Preview:')
st.write(data.head())

# Split features and target
X = data[feature_names]
Y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Sidebar for user input
st.sidebar.header('Adjust Input Parameters')

def user_input_features():
    # Adding interactive sliders for user input
    Architecture = st.sidebar.slider('Architecture Quality (1-5)', 1, 5, 3)
    Location = st.sidebar.slider('Location Potential (1-5)', 1, 5, 3)
    Land_Area = st.sidebar.slider('Land Area (m²)', float(X.Land_Area.min()), float(X.Land_Area.max()), float(X.Land_Area.mean()))
    Building_Area = st.sidebar.slider('Building Area (m²)', float(X.Building_Area.min()), float(X.Building_Area.max()), float(X.Building_Area.mean()))
    Bedrooms = st.sidebar.slider('Number of Bedrooms', 1, 5, 3)
    Bathrooms = st.sidebar.slider('Number of Bathrooms', 1, 5, 2)
    Living_Room = st.sidebar.slider('Number of Living Rooms', 1, 3, 1)
    Kitchen_Quality = st.sidebar.slider('Kitchen Quality (1-5)', 1, 5, 3)
    Security = st.sidebar.slider('Security Level (1-3)', 1, 3, 2)
    Greenness = st.sidebar.slider('Greenness of Area (1-3)', 1, 3, 2)
    Damage = st.sidebar.slider('Damage Level (1-5)', 1, 5, 1)
    House_Age = st.sidebar.slider('House Age (Years)', 0, 100, 10)
    Flood_Risk = st.sidebar.slider('Flood Risk Level (1-4)', 1, 4, 1)
    
    # Collecting user input into a dictionary
    user_data = {
        'Architecture': Architecture,
        'Location': Location,
        'Land_Area': Land_Area,
        'Building_Area': Building_Area,
        'Bedrooms': Bedrooms,
        'Bathrooms': Bathrooms,
        'Living_Room': Living_Room,
        'Kitchen_Quality': Kitchen_Quality,
        'Security': Security,
        'Greenness': Greenness,
        'Damage': Damage,
        'House_Age': House_Age,
        'Flood_Risk': Flood_Risk
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Display user input
st.write('### Your Selected Features:')
st.write(user_input)

# Train the model and make predictions
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(user_input)

# Manually format the predicted price into Rupiah (IDR) with thousands separators
formatted_price = "Rp {:,.0f}".format(prediction[0])

# Display predicted price
st.write('### Predicted Price of the House: ', formatted_price)

# Feature importance visualization
st.write('### Feature Importance:')
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
st.pyplot(fig)

# Model performance metrics
st.write('### Model Performance (on test data):')

# Make predictions on the test set
y_pred = model.predict(X_test)

# R-squared
score = model.score(X_test, y_test)
st.write(f'R-squared: {score:.2f}')

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error (MSE): {mse:.2f}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Explained Variance Score
evs = explained_variance_score(y_test, y_pred)
st.write(f'Explained Variance Score: {evs:.2f}')
