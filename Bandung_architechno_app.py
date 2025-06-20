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
X['Security'] *= 0.3
y = data["Price"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# --- Streamlit UI ---
st.title("🏠 Bandung House Price Prediction & Compatibility App")

# User input
st.sidebar.header("🏡 House Features")
def get_user_input():
    return pd.DataFrame([{
        'Architecture': st.sidebar.slider('Architecture Score', 100, 200, 150),
        'Location': st.sidebar.slider('Location Potential (1–5)', 1, 5, 3),
        'Land_Area': st.sidebar.slider('Land Area (m²)', 100, 500, 250),
        'Building_Area': st.sidebar.slider('Building Area (m²)', 50, 300, 150),
        'Bedrooms': st.sidebar.slider('Number of Bedrooms', 1, 6, 3),
        'Bathrooms': st.sidebar.slider('Number of Bathrooms', 1, 5, 2),
        'Living_Room': st.sidebar.slider('Living Rooms', 1, 3, 1),
        'Kitchen_Quality': st.sidebar.slider('Kitchen Quality (1–3)', 1, 3, 2),
        'Security': st.sidebar.slider('Security Level (0–3)', 0, 3, 1) * 0.3,
        'Greenness': st.sidebar.slider('Greenness of Area (0–3)', 0, 3, 2),
        'Damage': st.sidebar.slider('Damage Level (0–5)', 0, 5, 1),
        'House_Age': st.sidebar.slider('House Age (years)', 0, 100, 10),
        'Flood_Risk': st.sidebar.slider('Flood Risk Level (0–4)', 0, 4, 1),
    }])
user_input = get_user_input()

# User preference
st.sidebar.header("🎯 Your Preferences")
st.sidebar.markdown("Set the importance of each feature (1 = Not Important, 5 = Very Important)")
user_pref = {
    key: st.sidebar.slider(f"{key} Importance", 1, 5, 3)
    for key in X.columns
}
pref_series = pd.Series(user_pref)
pref_norm = pref_series / pref_series.sum()

# --- Prediction ---
user_input_scaled = scaler.transform(user_input)
predicted_price = model.predict(user_input_scaled)[0]

# Compatibility Score
user_scaled_df = pd.DataFrame(user_input_scaled, columns=X.columns)
def sigmoid(x): return 1 / (1 + np.exp(-x))
compat_score_raw = np.dot(user_scaled_df.values[0], pref_norm.values)
compatibility_score = sigmoid(compat_score_raw)

# --- Display Results ---
st.subheader("💰 Predicted House Price")
st.success(f"Rp {predicted_price:,.0f}")

st.subheader("🧩 Compatibility Score")
st.info(f"{compatibility_score * 100:.1f}% match")

# --- Feature Importance ---
st.subheader("📊 Feature Importance in Price Prediction")
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
st.pyplot(fig)

# --- Model Evaluation ---
st.subheader("📈 Model Evaluation Metrics")
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)

st.markdown(f"""
- **R-squared (R²)**: `{r2:.2f}` – Indicates how well the model explains the price variance.
- **Mean Absolute Error (MAE)**: `Rp {mae:,.0f}` – Average absolute prediction error.
- **Root Mean Squared Error (RMSE)**: `Rp {rmse:,.0f}` – Penalizes larger errors more.
- **Mean Absolute Percentage Error (MAPE)**: `{mape:.2f}%` – Error in percentage; lower is better.
- **Explained Variance Score**: `{evs:.2f}` – Measures explained variability of price.
""")

# --- Final Narrative Assessment ---
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
    match_level = "partial"
else:
    match_level = "low"

matched, mismatched = [], []
for feature in X.columns:
    val = user_input[feature].values[0]
    pref = pref_series[feature]
    mean_val = X[feature].mean()
    if pref >= 4 and val < mean_val:
        mismatched.append(feature)
    elif pref <= 2 and val > mean_val:
        mismatched.append(feature)
    else:
        matched.append(feature)

# Narrative generation
if price_level == "high" and match_level == "strong":
    message = "💰 The house is relatively expensive, but it's a great match for your preferences."
elif price_level == "high" and match_level == "low":
    message = "⚠️ The price is quite high, and it doesn't align well with what you care about."
elif price_level == "low" and match_level == "strong":
    message = "🎉 Affordable and matches your needs — this could be a great opportunity!"
elif price_level == "low" and match_level == "low":
    message = "🤔 Price is low, but compromises on your preferences may be needed."
elif price_level == "average" and match_level == "strong":
    message = "✅ The price is fair, and the house aligns well with your preferences."
elif price_level == "average" and match_level == "low":
    message = "📊 The price is standard, though some aspects might not match your expectations."
else:
    message = "🧠 This house is within normal range and partially meets your preferences."

st.info(message)
if matched:
    st.markdown(f"**✅ Features that match your preferences:** `{', '.join(matched)}`")
if mismatched:
    st.markdown(f"**❌ Features that might not meet your expectations:** `{', '.join(mismatched)}`")

st.caption("This conclusion is based on your feature importance settings and historical housing price trends.")

# --- Compatibility Ranking for All Houses ---

# 1. Pastikan preferensi valid
if 'user_pref' in locals() and isinstance(user_pref, dict) and len(user_pref) == X.shape[1]:

    # 2. Normalisasi preferensi
    pref_series = pd.Series(user_pref)
    pref_norm = pref_series / pref_series.sum()

    # 3. Hitung skor kecocokan
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    X_all_scaled = scaler.transform(X)

    # Pastikan dimensi sesuai
    if X_all_scaled.shape[1] == len(pref_norm):
        compat_raw_scores = np.dot(X_all_scaled, pref_norm.values)
        compat_scores = sigmoid(compat_raw_scores)

        # 4. Gabungkan ke dataset
        ranked_houses_df = data.copy()
        ranked_houses_df['Compatibility_Score'] = compat_scores
        ranked_houses_df['Score (%)'] = (compat_scores * 100).round(2)
        ranked_houses_df = ranked_houses_df.sort_values(by="Compatibility_Score", ascending=False)

        # 5. Cek data invalid
        if ranked_houses_df.isnull().values.any():
            st.warning("⚠️ Some values in the dataset are missing. Please check the data.")
            st.write(ranked_houses_df.isnull().sum())

        # 6. Tampilkan data maksimum 100 baris
        st.subheader("🏘️ Top Compatible Houses")
        st.markdown("Showing the top 100 most compatible houses based on your preferences.")
        st.dataframe(ranked_houses_df.reset_index(drop=True).head(100))

        # 7. Filter interaktif
        st.markdown("### 🔍 Filter Recommendations")
        max_price = st.slider("💰 Max Price (Rp)", int(data['Price'].min()), int(data['Price'].max()), int(data['Price'].max()))
        min_score = st.slider("⭐ Min Compatibility Score (%)", 0, 100, 70)

        filtered_df = ranked_houses_df[
            (ranked_houses_df['Price'] <= max_price) &
            (ranked_houses_df['Score (%)'] >= min_score)
        ]

        st.markdown("### ✅ Filtered Houses Based on Your Budget and Preference Score")
        st.dataframe(filtered_df.reset_index(drop=True).head(100))

    else:
        st.error("❌ Dimension mismatch between scaled features and preference weights.")

else:
    st.warning("⚠️ Preferences are not valid or incomplete. Please check the input sliders.")
