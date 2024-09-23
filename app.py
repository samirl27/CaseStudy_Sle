import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load the dataset and model training logic
def load_data():
    
    df = pd.read_csv('your_dataset.csv')
    
    df = pd.DataFrame({
        'area_type': ['Super built-up  Area', 'Built-up  Area', 'Plot  Area'],
        'availability': ['19-Dec', 'Ready To Move', '19-Jan'],
        'location': ['Electronic City Phase II', 'Whitefield', 'Sarjapur  Road'],
        'size': ['2 BHK', '3 BHK', '4 BHK'],
        'total_sqft': [1056, 1500, 2400],
        'bath': [2, 3, 4],
        'balcony': [1, 2, 3],
        'price': [39.07, 50.00, 70.00]
    })
    
    return df

def train_model():
    df = load_data()
    
    # Preprocessing: Drop missing values, one-hot encoding for categorical variables
    df = df.dropna()
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop(columns=['price'])
    y = df_encoded['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'decision_tree_model.pkl')
    
    return model

# Function to load the trained model
def load_model():
    model = joblib.load('decision_tree_model.pkl')
    return model

# Streamlit interface
def main():
    st.title("House Price Prediction using Decision Tree")
    
    # Sidebar inputs for the user to specify features
    area_type = st.selectbox('Area Type', ['Super built-up  Area', 'Built-up  Area', 'Plot  Area'])
    availability = st.selectbox('Availability', ['19-Dec', 'Ready To Move', '19-Jan'])
    location = st.selectbox('Location', ['Electronic City Phase II', 'Whitefield', 'Sarjapur  Road'])
    size = st.selectbox('Size', ['2 BHK', '3 BHK', '4 BHK'])
    total_sqft = st.number_input('Total Square Feet', min_value=300, max_value=10000, value=1000)
    bath = st.slider('Number of Bathrooms', 1, 5, 2)
    balcony = st.slider('Number of Balconies', 0, 3, 1)
    
    # Convert the inputs into a dataframe for prediction
    user_input = pd.DataFrame({
        'area_type': [area_type],
        'availability': [availability],
        'location': [location],
        'size': [size],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'balcony': [balcony]
    })
    
    # Preprocess the user input similarly to how the training data was preprocessed
    user_input_encoded = pd.get_dummies(user_input, drop_first=True)
    
    # Align with the training data (ensure all columns exist, even if user doesn't select them)
    df_encoded = pd.get_dummies(load_data(), drop_first=True)
    user_input_encoded = user_input_encoded.reindex(columns=df_encoded.columns, fill_value=0)
    
    # Load the trained model
    model = load_model()
    
    # Predict button
    if st.button('Predict Price'):
        prediction = model.predict(user_input_encoded)
        st.success(f"Predicted Price: {prediction[0]:.2f} Lakhs")
    
if __name__ == '__main__':
    # First, train the model (this could be skipped if the model is already saved)
    # Uncomment the line below if training is needed
    # train_model()
    
    main()
