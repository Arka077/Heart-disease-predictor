import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("Random_Forest.joblib")
    scaler = joblib.load("Scaler_Model.joblib")
    return model, scaler

# Function to make predictions
def predict_heart_disease(features, model, scaler):
    # Scale the features
    features_scaled = scaler.transform([features])
    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    return prediction[0], prediction_proba[0]

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Main title
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application predicts whether a patient has heart disease based on their medical attributes.
The model used is a Random Forest Classifier with 93.55% accuracy.
""")

try:
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Create a sidebar for additional information
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses a machine learning model to predict the likelihood of heart disease 
    based on several medical parameters. The model was trained on the Heart Disease UCI dataset.
    
    **Model:** Random Forest Classifier
    **Accuracy:** 93.55%
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Importance", "Dataset Info"])
    
    with tab1:
        st.header("Patient Information")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, value=45)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", 
                             options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                             index=0)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", 
                                  options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
        with col3:
            slope = st.selectbox("Slope of the Peak Exercise ST Segment", 
                               options=["Upsloping", "Flat", "Downsloping"])
            ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia", 
                              options=["Normal", "Fixed Defect", "Reversible Defect"])
        
        # Convert inputs to format expected by the model
        sex_encoded = 1 if sex == "Male" else 0
        cp_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        fbs_encoded = 1 if fbs == "Yes" else 0
        restecg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
        exang_encoded = 1 if exang == "Yes" else 0
        slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal_encoded = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1  # Assuming 1, 2, 3 encoding
        
        # Create feature array
        features = [age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                   restecg_encoded, thalach, exang_encoded, oldpeak, 
                   slope_encoded, ca, thal_encoded]
        
        # Prediction button
        if st.button("Predict"):
            prediction, prediction_proba = predict_heart_disease(features, model, scaler)
            
            # Display prediction
            st.header("Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("❗ **Heart Disease Detected**")
                else:
                    st.success("✅ **No Heart Disease Detected**")
                
                # Probability gauge
                prob_text = f"Probability of heart disease: {prediction_proba[1]:.2%}"
                st.markdown(f"**{prob_text}**")
                
                # Create a progress bar for probability
                st.progress(float(prediction_proba[1]))
            
            with col2:
                # Create a pie chart for the prediction probabilities
                fig, ax = plt.subplots(figsize=(4, 4))
                labels = ['No Disease', 'Heart Disease']
                sizes = [prediction_proba[0], prediction_proba[1]]
                colors = ['#3498db', '#e74c3c']
                explode = (0, 0.1)
                
                ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                      autopct='%1.1f%%', shadow=True, startangle=90)
                ax.axis('equal')
                
                st.pyplot(fig)
                
                # Display risk level
                risk_level = ""
                if prediction_proba[1] < 0.25:
                    risk_level = "Low Risk"
                elif prediction_proba[1] < 0.50:
                    risk_level = "Moderate Risk"
                elif prediction_proba[1] < 0.75:
                    risk_level = "High Risk"
                else:
                    risk_level = "Very High Risk"
                
                st.markdown(f"**Risk Level: {risk_level}**")
    
    with tab2:
        st.header("Feature Importance")
        # If feature importances are available in the model
        if hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                            'restecg', 'thalach', 'exang', 'oldpeak', 
                            'slope', 'ca', 'thal']
            
            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Display feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
            plt.title('Feature Importance for Heart Disease Prediction')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explain top features
            st.subheader("Top Influential Factors")
            for i in range(min(5, len(importance_df))):
                feature = importance_df.iloc[i]['Feature']
                importance = importance_df.iloc[i]['Importance']
                st.markdown(f"**{i+1}. {feature}** - Importance: {importance:.4f}")
        else:
            st.write("Feature importance not available for this model.")
    
    with tab3:
        st.header("Dataset Information")
        st.markdown("""
        ### Heart Disease Dataset
        
        This dataset contains 13 attributes that can be used to predict heart disease:
        
        1. **Age**: Age in years
        2. **Sex**: Gender (1 = male; 0 = female)
        3. **CP**: Chest pain type
           * 0: Typical angina
           * 1: Atypical angina
           * 2: Non-anginal pain
           * 3: Asymptomatic
        4. **Trestbps**: Resting blood pressure (in mm Hg)
        5. **Chol**: Serum cholesterol in mg/dl
        6. **FBS**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        7. **Restecg**: Resting electrocardiographic results
           * 0: Normal
           * 1: Having ST-T wave abnormality
           * 2: Showing probable or definite left ventricular hypertrophy
        8. **Thalach**: Maximum heart rate achieved
        9. **Exang**: Exercise induced angina (1 = yes; 0 = no)
        10. **Oldpeak**: ST depression induced by exercise relative to rest
        11. **Slope**: The slope of the peak exercise ST segment
            * 0: Upsloping
            * 1: Flat
            * 2: Downsloping
        12. **CA**: Number of major vessels (0-3) colored by fluoroscopy
        13. **Thal**: Thalassemia
            * 1: Normal
            * 2: Fixed defect
            * 3: Reversible defect
        
        **Target Variable**: Presence of heart disease (1 = yes, 0 = no)
        """)

except Exception as e:
    st.error(f"Error: {e}")
    st.warning("Please make sure 'Random_Forest.joblib' and 'Model.joblib' files are in the same directory as this app.")
    st.info("These files should contain the trained Random Forest model and StandardScaler respectively.")