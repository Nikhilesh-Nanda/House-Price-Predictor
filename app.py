import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="House Price Predictor", layout="wide")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.markdown(
    """
    This application predicts house prices based on various features using a 
    **Decision Tree Regressor** model.
    """
)
st.sidebar.markdown("---")
# --- Main App ---
st.title("🏡 House Price Predictor")
st.write("Enter the details of the house to get an estimated price.")

# --- Load Model ---
try:
    with open("model.pkl", "rb") as f:
        dt_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Preprocessing Function ---
def preprocess_inputs(input_data):
    # Get EXACT feature order used during training
    expected_columns = dt_model.feature_names_in_.tolist()

    # Create empty dataframe with correct column order
    processed_data = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Numerical features
    processed_data['area'] = input_data['area']
    processed_data['bedrooms'] = input_data['bedrooms']
    processed_data['bathrooms'] = input_data['bathrooms']
    processed_data['stories'] = input_data['stories']
    processed_data['parking'] = input_data['parking']
    processed_data['price_per_sqft'] = input_data['price_per_sqft']

    # Binary categorical (yes/no)
    processed_data['mainroad_yes'] = 1 if input_data['mainroad'] == 'yes' else 0
    processed_data['guestroom_yes'] = 1 if input_data['guestroom'] == 'yes' else 0
    processed_data['basement_yes'] = 1 if input_data['basement'] == 'yes' else 0
    processed_data['hotwaterheating_yes'] = 1 if input_data['hotwaterheating'] == 'yes' else 0
    processed_data['airconditioning_yes'] = 1 if input_data['airconditioning'] == 'yes' else 0
    processed_data['prefarea_yes'] = 1 if input_data['prefarea'] == 'yes' else 0

    # One-hot encoding for furnishingstatus (drop_first=True logic)
    if input_data['furnishingstatus'] == 'semi-furnished':
        processed_data['furnishingstatus_semi-furnished'] = 1
    elif input_data['furnishingstatus'] == 'unfurnished':
        processed_data['furnishingstatus_unfurnished'] = 1
    # 'furnished' remains base case (both 0)

    return processed_data


# --- Input Form ---
with st.form("house_features_form"):
    st.header("House Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("Area (in sqft)", 1500, 17000, 5000, 100)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 1)
        stories = st.slider("Stories", 1, 4, 2)
        parking = st.slider("Parking Spaces", 0, 3, 1)
        price_per_sqft = st.number_input(
            "Price per sqft (Scaled: Price/1000/Area)",
            min_value=0.40,
            max_value=3.00,
            value=1.00,
            step=0.01
        )

    with col2:
        mainroad = st.selectbox("Proximity to Main Road", ['yes', 'no'])
        guestroom = st.selectbox("Has Guest Room", ['yes', 'no'])
        basement = st.selectbox("Has Basement", ['yes', 'no'])
        hotwaterheating = st.selectbox("Has Hot Water Heating", ['yes', 'no'])
        airconditioning = st.selectbox("Has Air Conditioning", ['yes', 'no'])
        prefarea = st.selectbox("Is in a Preferred Area", ['yes', 'no'])
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ['furnished', 'semi-furnished', 'unfurnished']
        )

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        user_inputs = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus,
            'price_per_sqft': price_per_sqft
        }

        processed_input_df = preprocess_inputs(user_inputs)

        try:
            predicted_price_scaled = dt_model.predict(processed_input_df)[0]
            predicted_price = predicted_price_scaled * 1000

            st.success(f"Estimated House Price: **₹ {predicted_price:,.2f}**")
            st.info("Note: Model predicts price in thousands (₹'000) internally.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

