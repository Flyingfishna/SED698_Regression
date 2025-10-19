import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

st.title("📈 Advertising Sales Prediction App")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload advertising.csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.write(data.head())

    # Step 2: Split data into features (X) and target (y)
    X = data.drop("sales", axis=1)
    y = data["sales"]

    # Step 3: Build and train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Step 4: Save model
    with open("model-reg-xxx.pkl", "wb") as file:
        pickle.dump(model, file)

    st.success("✅ Model trained and saved as 'model-reg-xxx.pkl'")

    # Step 5: Make prediction for user input
    st.subheader("Make a New Prediction")

    youtube = st.number_input("YouTube budget", value=50.0)
    tiktok = st.number_input("TikTok budget", value=50.0)
    instagram = st.number_input("Instagram budget", value=50.0)

    new_data = pd.DataFrame({
        "youtube": [youtube],
        "tiktok": [tiktok],
        "instagram": [instagram]
    })

    predicted_sales = model.predict(new_data)
    st.write(f"### 💰 Estimated Sales: {predicted_sales[0]:.2f}")
