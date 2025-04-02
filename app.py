# our team - working
import gdown  
import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load models (cached)
@st.cache_resource
def load_models():
    # Download files from Google Drive
    
    gdown.download(
        url="https://drive.google.com/uc?id=1GmKSsV98X3k5vNSQPmPtRErIYiA1t9r6",  # Replace with your ID
        output="rf_model.joblib",
        quiet=False
    )
    gdown.download(
        url="https://drive.google.com/uc?id=1qU_4mzVdWyadC_JMGF1ow2CfrVODIwdX",  # Replace with your ID
        output="crop_yield_model.h5",
        quiet=False
    )
    gdown.download(
        url="https://drive.google.com/uc?id=1zuKL1JQPHjVsQY9TBeHLpOxKi3PbDhRx",  # Replace with your ID
        output="crop_yield_scaler.joblib",
        quiet=False
    )

    try:
        rf_model = joblib.load('rf_model.joblib')
        try:
            dnn_model = load_model('crop_yield_model.h5')
        except:
            dnn_model = load_model('crop_yield_model.keras')
        scaler = joblib.load('crop_yield_scaler.joblib')
        return rf_model, dnn_model, scaler
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        st.stop()

rf_model, dnn_model, scaler = load_models()

def predict(input_data, model_type):
    try:
        if model_type == "Random Forest":
            return rf_model.predict(input_data.reshape(1, -1))[0]
        else:
            scaled_input = scaler.transform(input_data.reshape(1, -1))
            return dnn_model.predict(scaled_input)[0][0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Sidebar navigation
st.sidebar.title("Explore")
page = st.sidebar.radio("Menu", ["Our Project", "Predictor", "Our Team"])

if page == "Predictor":
    # Main prediction interface
    st.title("🌾 Crop Yield Prediction")
    model_type = st.sidebar.radio("Select Model", ("Random Forest", "Deep Neural Network"))

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", 2000, 2030, 2023)
            ndvi = st.slider("NDVI", 0.0, 1.0, 0.5)
            evi = st.slider("EVI", 0.0, 1.0, 0.3)
            gpp = st.number_input("GPP", 0.0, 500.0, 200.0)
            soil = st.slider("Soil Moisture", 0.0, 1.0, 0.3)
        with col2:
            lst = st.number_input("Land Surface Temp (°C)", 0.0, 50.0, 30.0)
            rain = st.number_input("Precipitation (mm)", 0.0, 20.0, 5.0)
            area = st.number_input("Area (Hectares)", 0, 500000, 100000)
            production = st.number_input("Production (Tonnes)", 0, 1000000, 200000)
        
        if st.form_submit_button("Predict Yield"):
            input_data = np.array([year, ndvi, evi, gpp, soil, lst, rain, area, production])
            prediction = predict(input_data, model_type)
            
            st.success(f"""
            ## Prediction Result
            **Model:** {model_type}  
            **Predicted Yield:** {prediction:.1f} Tonnes/Hectare
            """)
            st.balloons()

elif page == "Our Project":
    # Our Project page
    st.title("Our Project")
    st.header("AI Driven Hyperlocal Dynamic Machine Learning Model To Predict Crop Yield")
    
    # Problem Statement
    st.header("Problem Statement")
    st.write("""
    Agriculture is a vital sector that sustains the global population by providing food and raw materials. 
    However, modern agricultural systems face increasing challenges due to growing population 
    pressures, climate change, and the need for sustainable resource management. One of the major 
    challenges for farmers and agricultural stakeholders is predicting crop yields accurately, a task that is 
    crucial for optimizing resource allocation and ensuring food security. Traditional methods of yield 
    prediction, which rely on historical data or general weather patterns, are often inadequate in the face 
    of rapidly changing environmental conditions and localized farming practices. The lack of precision 
    in these predictions leads to inefficiencies in crop management, resource use, and ultimately, 
    agricultural productivity. """)
    
    # Objectives
    st.header("Objectives")
    st.write("""
    1. To analyze remote sensing data to improve yield prediction accuracy and capture critical environmental and crop health factors like NDVI & EVI.
    2. To develop a highly accurate machine learning model that predicts crop yield by leveraging both spatial and temporal features extracted from satellite data.
    3. To study groundnut crop yield across five districts of Gujarat using real-time data from MODIS satellite.
    """)
    
    # System Architecture (without border)
    st.header("System Architecture")
    try:
        st.image("System_Architecture.png")
    except Exception as e:
        st.warning(f"System architecture image not found. Error: {str(e)}")

    # Video Section with Button that opens in new tab
    st.header("Project Demo")
    st.image("video.png")
    
    video_url = "https://drive.google.com/file/d/1MTLHa3f7b8dQaIgXUYzf2ppNHIAGjlbX/view?usp=drive_link"
    
    # Using HTML to ensure new tab opens
    st.markdown(f"""
    <div style="text-align: center;">
        <a href="{video_url}" target="_blank">
            <button style="background-color: #4CAF50; color: white; border: none;
                        padding: 10px 20px; text-align: center; display: inline-block;
                        font-size: 16px; margin: 10px 0; cursor: pointer;
                        border-radius: 8px;">
                Watch Project Demo Video
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
else:
    # Our Team page
    st.title("Our Team")
    st.header("Meet the Team Behind the Project")
    
    # Custom CSS for team cards
    st.markdown("""
    <style>
    
    .team-container {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 20px;
    }
                
    .team-card {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        padding: 20px;
        margin: 15px 0;
        background-color: #f9f9f9;
        transition: 0.3s;
        width : 250px
                
    }
    .team-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .team-name {
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 5px;
    }
    .team-role {
        font-style: italic;
        color: #555;
        margin-bottom: 10px;
    }
    .linkedin-btn {
        background-color: #0077b5;
        color: white !important;
        padding: 5px 10px;
        text-decoration: none;
        border-radius: 5px;
        display: inline-block;
        margin-top: 10px;
        font-size: 14px;
    }
    .linkedin-btn:hover {
        background-color: #005f8e;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Team members data
    team_members = [
        # {
        #     "name": "Ms. Priya Kaul",
        #     "role": "Project Guide",
        #     "linkedin": "https://linkedin.com/in/johndoe"
        # },
        {
            "name": "Srushti Kale",
            "role": "Team Member",
            "linkedin": "https://www.linkedin.com/in/srushtikale07/"
        },
        {
            "name": "Manish Patil",
            "role": "Team Member",
            "linkedin": "https://www.linkedin.com/in/manish-patil-687356248/"
        },
        {
            "name": "Shweta Nadar",
            "role": "Team Member",
            "linkedin": "https://www.linkedin.com/in/shwetanadar/"
        }
    ]
    
    # Project guide
    st.markdown(f"""
        
        <div class="team-card">
            <div class="team-name">{"Ms. Priya Kaul"}</div>
            <div class="team-role">{"Project Guide"}</div>
            <a href="{"https://www.linkedin.com/in/priya-kaul/"}" target="_blank" class="linkedin-btn">
                LinkedIn Profile
            </a>
        </div>
        
        """, unsafe_allow_html=True)
    
    # Display all cards in a single row
    st.markdown('<div class="team-container">', unsafe_allow_html=True)

    # Display team members
    for member in team_members:
        st.markdown(f"""
        <div class="team-card">
            <div class="team-name">{member['name']}</div>
            <div class="team-role">{member['role']}</div>
            <a href="{member['linkedin']}" target="_blank" class="linkedin-btn">
                LinkedIn Profile
            </a>
        </div>
        
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)





    # # Team photo (optional)
    # try:
    #     st.header("Team Photo")
    #     st.image("team_photo.jpg", use_container_width=True, caption="Our amazing team working together")
    # except:
    #     pass















































































































