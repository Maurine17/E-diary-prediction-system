import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Function to load dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to prepare the model
def prepare_model(data):
    X = data.drop(columns=['MilkProduction'])
    y = data['MilkProduction']
    
    numerical_features = ['Age', 'Weight', 'LactationPeriod', 'FeedAmount', 'WaterIntake', 
                          'Temperature', 'Humidity', 'PastMilkProduction', 'MilkingFrequency', 'VetVisits']
    categorical_features = ['Breed', 'HealthStatus', 'FeedType', 'BarnCondition']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model.fit(X_train, y_train)
    print(f'Model training score: {model.score(X_train, y_train)}')
    
    return model

# Function to predict milk production
def predict_milk_production(model, input_data):
    input_df = pd.DataFrame([input_data])
    return model.predict(input_df)[0]

# Function to provide advice based on input data
def provide_advice(input_data):
    advice = []
    if input_data['HealthStatus'] == 'Sick':
        advice.append("Advice: Your cow is sick. Consult a veterinarian for appropriate medical treatment.")
    if input_data['FeedAmount'] < 20:
        advice.append("Advice: Increase the feed amount to at least 20 kg per day to ensure adequate nutrition.")
    if input_data['WaterIntake'] < 40:
        advice.append("Advice: Increase the water intake to at least 40 liters per day to ensure proper hydration.")
    if input_data['Temperature'] > 30:
        advice.append("Advice: Ensure proper ventilation and cooling to prevent heat stress.")
    if input_data['Humidity'] > 80:
        advice.append("Advice: Ensure proper ventilation to prevent respiratory problems.")
    if input_data['BarnCondition'] == 'Dirty':
        advice.append("Advice: Clean the barn regularly to prevent diseases and infections.")
    if input_data['MilkingFrequency'] < 2:
        advice.append("Advice: Increase the milking frequency to at least twice a day for optimal milk production.")
    if input_data['VetVisits'] < 4:
        advice.append("Advice: Schedule regular veterinary visits to ensure the health and well-being of your cow.")
            
    return advice

# Streamlit app
def main():
    st.title("Dairy Farm Management System")

    # Load the dataset
    file_path = "./data/dataset_1.csv"
    data = load_dataset(file_path)
    print(data.head())
    # Prepare the model
    model = prepare_model(data)

    st.header("Enter the following details for your animal:")
    input_data = {
        'AnimalID': st.number_input("Animal ID", min_value=1, max_value=100, step=1),
        'Breed': st.selectbox("Breed", ['BreedA', 'BreedB', 'BreedC']),
        'Age': st.number_input("Age in years", min_value=1, max_value=20, step=1),
        'Weight': st.number_input("Weight in kg", min_value=100, max_value=1000, step=1),
        'LactationPeriod': st.number_input("Lactation Period in days", min_value=100, max_value=400, step=1),
        'HealthStatus': st.selectbox("Health Status", ['Healthy', 'Sick']),
        'FeedType': st.selectbox("Feed Type", ['TypeA', 'TypeB', 'TypeC']),
        'FeedAmount': st.number_input("Feed Amount in kg per day", min_value=0.0, max_value=50.0, step=0.1),
        'WaterIntake': st.number_input("Water Intake in liters per day", min_value=0.0, max_value=200.0, step=0.1),
        'Temperature': st.number_input("Temperature in degree Celsius", min_value=-10.0, max_value=50.0, step=0.1),
        'Humidity': st.number_input("Humidity in percentage", min_value=0.0, max_value=100.0, step=0.1),
        'BarnCondition': st.selectbox("Barn Condition", ['Clean', 'Moderate', 'Dirty']),
        'PastMilkProduction': st.number_input("Past Milk Production in liters per day", min_value=0.0, max_value=50.0, step=0.1),
        'MilkingFrequency': st.number_input("Milking Frequency per day", min_value=0, max_value=10, step=1),
        'VetVisits': st.number_input("Veterinary Visits per month", min_value=0, max_value=10, step=1)
    }

    if st.button("Predict Milk Production"):
        # Predict milk production
        predicted_milk_production = predict_milk_production(model, input_data)
        # This prediction is based on the input data provided by the user if the user provides the correct data then the prediction will be correct
        st.header("Milk Production Prediction")
        st.success(f"""
                    Based on the input data provided, the model predicts the milk production of the animal.
                    The animal is expected to produce {predicted_milk_production:.2f} liters of milk per day.
                   """)

        # Provide advice
        advice = provide_advice(input_data)
        for item in advice:
            st.warning(item)

if __name__ == "__main__":
    main()
