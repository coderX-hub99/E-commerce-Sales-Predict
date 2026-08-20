import streamlit as st
import pandas as pd
import numpy as np
import joblib


model=joblib.load("sales_model.pkl")
scaler=joblib.load("scaler.pkl")
feature_names=joblib.load("feature_names.pkl")


st.title(" 🛍️ E-commerce Sales Prediction ")
st.write("Predict the Discounted Price of Product from category")
st.subheader("Product Information")

col1,col2=st.columns(2)
with col1:
 quantity_sold=st.number_input("quantity_sold",min_value=1,max_value=5)
 Unit_Price=st.number_input("Unit_Price",min_value=28,max_value=392)
 Customer_Age=st.number_input("Customer_Age",min_value=2,max_value=100)
 Review_Rating=st.number_input("Review_Rating",min_value=1,max_value=5)
with col2:
 Shipping_Cost=st.number_input("Shipping_Cost",min_value=1,max_value=50)
 Profit_Margin=st.number_input("Profit_Margin",min_value=1,max_value=50)
 Product_Name=st.selectbox("Product_Name",["Smartphone","Yoga Mat","T-Shirt","Lipstick","Coffee Maker"])
 Product_Category=st.selectbox("Product_Category",["Clothing ","Electronics","Sports","Beauty","Home & Kitchen"])


input=pd.DataFrame({
    "quantity_sold":[quantity_sold],
    "Unit_Price":[Unit_Price],
    "Customer_Age":[Customer_Age],
    "Review_Rating":[Review_Rating],
    "Shipping_Cost":[Shipping_Cost],
    "Profit_Margin":[Profit_Margin],
    "Product_Name":[Product_Name],
    "Product_Category":[Product_Category],
})

st.subheader("Model Performance")
col1,col2,col3=st.columns(3)
with col1:st.metric("R2",0.879)
with col2:st.metric("MAE",104.9)
with col3:st.metric("RMSE",182.9)

st.divider()

input_encode=pd.get_dummies(input[["Product_Name","Product_Category"]])
input_final=pd.concat([input[["quantity_sold","Unit_Price",
                              "Customer_Age","Review_Rating",
                              "Shipping_Cost","Profit_Margin"]],input_encode],axis=1)
input_final=input_final.reindex(columns=feature_names,fill_value=0)
input_scaler=scaler.transform(input_final)
if st.button("Predict Discounted Price"):
    predict=model.predict(input_scaler)[0]
    st.info("💡 Busniss Insight" \
    "The Model estimates the expected discounted prices based on" \
    " the product, customer and sales characterstics provided")
    st.success(f"🎯 Predicted Discounted Price : ₹{predict :.2f}")
st.write("Built with Python, Scikit-learn & Streamlit")

  
