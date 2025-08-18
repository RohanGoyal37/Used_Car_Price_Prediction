import streamlit as st
import numpy as np
import pandas as pd
from Model import model
from prepro import load_data


def app(df):
    st.header("This app uses Extreme Gradient Boosting to predict the price of a used car based on the inputs.")
    st.subheader("Select values:")

    year_options = [1983, 1991, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    years = st.selectbox("Year", year_options, index=year_options.index(2015) if 2015 in year_options else 0)
    km_driven = st.number_input("KM driven", min_value=0, max_value=600000, value=50000, step=1000)
    fuel_options = ['CNG', 'Diesel', 'LPG', 'Petrol']
    fuel = st.selectbox("Fuel", fuel_options, index=fuel_options.index('Petrol'))
    seller_type_options = ['Dealer', 'Individual', 'Trustmark Dealer']
    seller_type = st.selectbox("Seller Type", seller_type_options, index=seller_type_options.index('Individual'))
    transmission_options = ['Automatic', 'Manual']
    transmission = st.selectbox("Transmission", transmission_options, index=transmission_options.index('Manual'))
    owner_options = ['First Owner', 'Fourth & Above Owner', 'Second Owner', 'Test Drive Car', 'Third Owner']
    owner = st.selectbox("Owner", owner_options, index=owner_options.index('First Owner'))
    mileage = st.number_input("Mileage", min_value=10.0, max_value=50.0, value=20.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=600.0, max_value=5000.0, value=1200.0, step=1.0)
    max_power = st.number_input("Max Power", min_value=20.0, max_value=450.0, value=80.0, step=0.1)
    seats_options = [2, 4, 5, 6, 7, 8, 9, 10, 14]
    seats = st.selectbox("Seats", seats_options, index=seats_options.index(5))
    brand_list = ['Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    brand = st.selectbox("Brand", brand_list, index=brand_list.index("Maruti") if "Maruti" in brand_list else 0)

    # Show record count for selected brand and warning if rare
    df_full = pd.read_csv('Car details v3.csv')
    selected_brand_count = df_full['name'].str.split().str[0].value_counts().get(brand, 0)
    st.info(f"Records for selected brand ({brand}): {selected_brand_count}")
    if selected_brand_count < 20:
        st.warning(f"Warning: Only {selected_brand_count} records for {brand} in the dataset. Prediction may be less reliable.")

    # Encode categorical variables as in training
    # You may want to use the same LabelEncoder mapping as in training for production
    # For now, use the same order as in the training data
    fuel_map = {v: i for i, v in enumerate(sorted(["Petrol", "Diesel", "CNG", "LPG", "Electric"]))}
    seller_type_map = {v: i for i, v in enumerate(sorted(["Individual", "Dealer", "Trustmark Dealer"]))}
    transmission_map = {"Manual": 1, "Automatic": 0}
    owner_map = {"First Owner": 0, "Second Owner": 2, "Third Owner": 4, "Fourth & Above Owner": 1, "Test Drive Car": 3}

    feature_dict = {
        'year': [years],
        'km_driven': [km_driven],
        'fuel': [fuel_map.get(fuel, 0)],
        'seller_type': [seller_type_map.get(seller_type, 0)],
        'transmission': [transmission_map.get(transmission, 1)],
        'owner': [owner_map.get(owner, 0)],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'brand': [0],  # Default to 0, or you can build a brand_map as above
    }
    feature = pd.DataFrame(feature_dict)
    if (st.button("Predict")):

        acc,pred_price,rsquare_score, mae, msle, rmse = model(df,feature)[1:]
        
        st.success(f"Predicted Price Of Car Rs {round(pred_price)}")
        st.balloons()
        st.info(f"Accuracy of Our Model is {round(acc, 2)}")
        st.info(f"R-squared score of this model is: {rsquare_score:.2}")
        st.info(f"Mean absolute error of this model is: {mae:.3f}")
        if msle is not None:
            st.info(f"Mean squared log error of this model is: {msle:.3f}")
        else:
            st.warning("Mean squared log error (MSLE) is not available due to negative values in predictions or targets.")
        st.info(f"Root mean squared error of this model is: {rmse:.3f}")