import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page layout and title
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load and cache the pre-trained KNN Regressor model."""
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the root directory.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()

# Header Section
st.title("🎓 Student Performance Predictor")
st.markdown("Use this machine learning application to estimate student performance based on enrolled courses and study duration.")
st.divider()

# Sidebar Setup
st.sidebar.header("⚙️ Configuration & Info")
st.sidebar.info(
    "**Model Type:** K-Neighbors Regressor\n\n"
    "**Features Required:**\n"
    "- Number of Courses\n"
    "- Time Spent Studying (Hours)\n\n"
    "Fill in the inputs on the main panel to view predictions."
)

if model is not None:
    # Main Input Layout (Two Column Setup)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 Input Parameters")
        
        number_courses = st.number_input(
            "Number of Enrolled Courses (`number_courses`)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Select the total number of courses the student is taking."
        )

        time_study = st.number_input(
            "Study Time in Hours (`time_study`)",
            min_value=0.0,
            max_value=24.0,
            value=4.5,
            step=0.5,
            help="Select the total hours per day/week spent on studying."
        )

        predict_btn = st.button("🚀 Calculate Prediction")

    with col2:
        st.subheader("📊 Output & Analytics")

        if predict_btn:
            # Build input DataFrame matching exact feature names
            input_df = pd.DataFrame(
                [[number_courses, time_study]], 
                columns=["number_courses", "time_study"]
            )

            try:
                # Perform inference
                prediction = model.predict(input_df)[0]

                # Result Display
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Predicted Score / Performance Indicator",
                    value=f"{prediction:.2f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                st.success("Prediction generated successfully!")
                
                with st.expander("🔍 View Raw Model Input"):
                    st.dataframe(input_df)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("Adjust the parameters on the left and click **Calculate Prediction**.")
