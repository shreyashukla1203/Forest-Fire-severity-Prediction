import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt

#loading the saved model
model=pickle.load(open('model.sav','rb'))

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64("img.jpeg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.pexels.com/photos/2509093/pexels-photo-2509093.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500");
    background-size: 100%;
    background-color: black;
    color: white;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}
[data-testid="stSidebar"] {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}
[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


def predict_severity(feature_list):
        
    # Create an array of zeros with shape (1, 37)
    input_features = np.zeros((1, 37))
    
    # Set the values of the input features to the user input
    input_features[0, [2, 5, 8, 10, 15, 17, 21, 22]] = feature_list
    
    # Make the prediction
    prediction = model.predict(input_features)
    
    # Return the predicted severity level
    return prediction[0]

st.title('Forest Fire Severity Prediction')
st.write("Forest fires is one of the important catastrophic events and have great impact on environment, infrastructure and human life. This application aims to predict the burned area of the forest fires specifically in the northeast region of Portugal-Montesinho park, based on the spatial, temporal and weather variables wusing RandomForestRegressor.We use public dataset from UCI Machine Learning Repository")
st.subheader('Predict the severity of a forest fire based on various features')
st.write('Enter the values for the features in the sidebar to make a prediction:')


# Input fields for the important features
feature_names = ['FFMC (Fine Fuel Moisture Code)', 'DMC (Duff Moisture Code)', 'DC (Drought Code)', 'ISI (Initial Spread Index)', 'temp', 'RH', 'wind', 'rain']
feature_ranges = [(min_value, max_value) for min_value, max_value in [
    # Update with the actual range for each feature
    (18.7,96.20),  # FFMC
    (1.1,291.3),  # DMC
    (7.9,860.6),  # DC
    (0.0,56.10),  # ISI
    (2.2,33.30),  # temp
    (15, 100),  # RH
    (0.40,9.40),  # wind
    (0.0,6.4)  # rain
]]

feature_values = []
for feature_name, feature_range in zip(feature_names, feature_ranges):
    # Create a slider for each feature with the specified range
    value = st.sidebar.slider(feature_name, feature_range[0], feature_range[1])
    feature_values.append(value)

# Button to make the prediction
if st.sidebar.button('Predict'):
    # Call the predict_severity() function to make the prediction
    prediction = predict_severity(feature_values)
    predicted_area = np.exp(prediction)
    pred_area=round(predicted_area,2)
    # Display the predicted severity level
    st.write('The predicted area burnt is:', pred_area, 'hectares')
    
    st.info("The area is measured in hectares, which is a unit of land area equal to 10,000 square meters or approximately 2.47 acres.")
    # Convert the predicted area from hectares to acres
    predicted_area_acres = round((pred_area * 2.47105) ,2)

    # Convert predicted_area_acres to string and concatenate it with the message
    message = 'Therefore, ' + str(predicted_area_acres) + ' acres of land has been burnt in the forest fire!'

    # Display the message using st.info()
    st.info(message)
   
if st.button("Click to see visualizations"):
    
    st.write("##### Visualization of the importance of features in terms of effect on the predicting ability of the model.")
    st.image('featureimpgraph.png')
    st.error("Obseravation: We can see 'temp_F' feature has highest importance followed by 'DMC'")
    # Visualization: Bar chart - Frequency of forest fires by month

    data = pd.read_csv('forestfires.csv')

    st.write("##### Visualization of frequency of forest fire by month")
    data['month'] = pd.to_datetime(data['month'], format='%b').dt.month
    month_counts = data['month'].value_counts().sort_index()
    month_names = pd.to_datetime(month_counts.index, format='%m').strftime('%b')

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(month_names, month_counts, color='green')  # Set the color to green
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Frequency of Forest Fires by Month')
    st.pyplot(fig1)
    st.error("Observation: The month of August and September had the largest forest fires in terms of burnt areas !")

    st.write("##### Visualization of fire count for each day")
    st.image("firecount.png")
    st.error("Obseravtion: Majority of the forest fires occured during days when there was less working activity (Sunday, Friday and Saturday). This could imply that forest fires are most likely to happen on week-ends than week days.")



# Custom CSS style for the disclaimer message
disclaimer_style = """
    <style>
    .disclaimer {
        background-color: #f8f8f8;
        color: #ff0000;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
"""

# Display the disclaimer message with custom style
st.markdown(disclaimer_style, unsafe_allow_html=True)
st.markdown('<div class="disclaimer">**Note: This forest fire prediction app is a state-of-the-art model, but it is not perfect. It is important to use the app in conjunction with other sources of information to make informed decisions about forest fire safety.</div>', unsafe_allow_html=True)




