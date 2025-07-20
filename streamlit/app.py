import streamlit as st
import requests
import os

api_url = os.getenv("FASTAPI_URL")


st.set_page_config(page_title="Price Optimizer", layout="centered")

st.title("💰 Pricing Conversion Predictor")

# === Input form ===
with st.form("prediction_form"):
    offered_price = st.number_input("Offered Price", min_value=0.0, format="%.2f")
    base_price = st.number_input("Base Price", min_value=0.0, format="%.2f")
    competitor_price = st.number_input("Competitor Price", min_value=0.0, format="%.2f")

    category = st.selectbox("Product Category", ["electronics", "books", "clothing", "home", "beauty"])
    segment = st.selectbox("Customer Segment", ["deal-seeker", "premium", "loyal", "new", "impulsive"])
    time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    season = st.selectbox("Season", ["peak", "off"])

    submit = st.form_submit_button("Predict")

# === Send request to FastAPI ===
if submit:
    payload = {
        "offered_price": offered_price,
        "base_price": base_price,
        "competitor_price": competitor_price,
        "category": category,
        "segment": segment,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "season": season,
    }

    with st.spinner("Predicting..."):
        try:
            
            response = requests.post(api_url, json=payload)

            if response.status_code == 200:
                result = response.json()
                if "converted_probability" in result:
                    st.success(f"🟢 Probability of Conversion: {result['converted_probability'] * 100:.2f}%")
                else:
                    st.error(f"API Error: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API Request Failed: {response.status_code}")

        except Exception as e:
            st.error(f"❌ Exception: {str(e)}")
