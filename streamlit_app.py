import streamlit as st
import requests
import joblib

# Load feature list
try:
    features = joblib.load('features.pkl')
except:
    st.error("Model not trained yet")
    st.stop()

st.set_page_config(page_title="COVID-19 Predictor")
st.title("COVID-19 Test Prediction")

# Create form
with st.form("covid_form"):

    # Ask for age and sex
    st.write("#### Personal Information")
    # Age input
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    # Sex input
    sex = st.radio("Sex", ["Male", "Female"])

    st.divider()
    
    # Symptoms checkboxes
    st.write("#### Symptoms Checklist")
    symptoms = {}
    for feature in features:
        if feature not in ['Age', 'Sex']:
            symptoms[feature] = st.checkbox(feature.replace('_', ' ').title())
    
    submitted = st.form_submit_button("Get Prediction")

if submitted:
    # Prepare data for API
    data = {
        'Age': age,
        'Sex': 1 if sex == "Male" else 0,
        **{k: 1 if v else 0 for k, v in symptoms.items()}
    }
    with st.spinner("Analyzing symptoms..."):
        
        # Call API
        try:
            response = requests.post("http://localhost:5000/predict", json=data)
            if response.status_code == 200:
                result = response.json()['result']
                st.write("### Prediction Result")
                if result == "POSITIVE":
                    st.error("POSITIVE")
                else:
                    st.success("NEGATIVE")
            else:
                st.error("Prediction failed")
        except:
            st.error("Could not connect to prediction service")