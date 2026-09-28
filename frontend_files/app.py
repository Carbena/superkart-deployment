import streamlit as st
import pandas as pd
import requests

# Backend URL — 'backend' is the container name on the shared Docker network
BACKEND_URL = "http://backend:5000"

st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")
st.title("SuperKart Sales Forecasting")
st.write("Predict product sales revenue for SuperKart outlets.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- SINGLE PREDICTION ----------------
with tab1:
    st.header("Single Product Prediction")

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.5)
        product_sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Allocated Area (ratio)", min_value=0.001, max_value=0.5, value=0.05)
        product_mrp = st.number_input("Product MRP", min_value=30.0, max_value=300.0, value=147.0)
        product_id_char = st.selectbox("Product Category Code", ["FD", "DR", "NC"])

    with col2:
        store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1",
                                                  "Supermarket Type2", "Food Mart"])
        store_age = st.number_input("Store Age (years)", min_value=0, max_value=50, value=10)
        product_type_cat = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_type_cat
        }

        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Sales: {result['predicted_sales']}")
            else:
                st.error(f"Error: {response.json().get('error')}")
        except Exception as e:
            st.error(f"Could not reach the backend: {e}")

# ---------------- BATCH PREDICTION ----------------
with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file with the required columns to get predictions for multiple records.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_preview.head())

        if st.button("Run Batch Prediction"):
            uploaded_file.seek(0)
            try:
                response = requests.post(
                    f"{BACKEND_URL}/batch_predict",
                    files={"file": uploaded_file}
                )
                if response.status_code == 200:
                    results = pd.DataFrame(response.json())
                    st.success(f"Predictions generated for {len(results)} records.")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                else:
                    st.error(f"Error: {response.json().get('error')}")
            except Exception as e:
                st.error(f"Could not reach the backend: {e}")
