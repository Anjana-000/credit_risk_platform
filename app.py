import streamlit as st
import pandas as pd
import os
from src.ml.predict import predict_risk
from src.talk_to_data.nl_to_sql import ask_data
from src.utils.config import MODEL_PATH
import joblib

st.set_page_config(page_title="NeoStats Credit Risk Platform", layout="wide")

st.title("🏦 AI-Powered Credit Risk Intelligence Platform")
st.markdown("Developed for NeoStats | Intelligence. Innovation. Impact.")

tab1, tab2, tab3 = st.tabs(["📊 Data & EDA", "🔮 Risk Prediction & XAI", "💬 Talk-to-Data Chatbot"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.write("Overview of the Home Credit Default Risk Dataset.")
    
    
    # Safely check if images exist before asking Streamlit to draw them
    if os.path.exists("data/target_dist.png") and os.path.exists("data/income_credit.png"):
        st.image("data/target_dist.png", width=500)
        st.image("data/income_credit.png", width=600)
    else:
        st.warning("Visualizations not found. Please run `python notebooks/eda.py` in your terminal to generate them.")

with tab2:
    st.header("Applicant Risk Prediction")
    
    if not MODEL_PATH.exists():
        st.error("Model not trained yet. Please run `python -m src.ml.train` first.")
    else:
        # Dynamic form based on model features
        pipeline = joblib.load(MODEL_PATH)
        features = pipeline.feature_names_in_
        
        st.write("Enter Applicant Financials:")
        col1, col2 = st.columns(2)
        
        input_data = {}
        for i, feat in enumerate(features[:10]): # Limit to top 10 for UI cleanliness
            with (col1 if i % 2 == 0 else col2):
                input_data[feat] = st.number_input(f"{feat}", value=0.0)
                
        # Fill remaining required features with medians
        for feat in features[10:]:
            input_data[feat] = 0.0

        if st.button("Assess Credit Risk"):
            input_df = pd.DataFrame([input_data])
            proba, band, shap_df = predict_risk(input_df)
            
            st.markdown("### Results")
            st.metric(label="Probability of Default", value=f"{proba*100:.2f}%")
            
            if band == "Low Risk":
                st.success(f"Risk Band: {band}")
            elif band == "Medium Risk":
                st.warning(f"Risk Band: {band}")
            else:
                st.error(f"Risk Band: {band}")
                
            st.markdown("### Explainable AI (SHAP)")
            st.write("Top features driving this specific decision:")
            st.dataframe(shap_df.head(5))

with tab3:
    st.header("Talk-to-Data")
    st.write("Ask natural language questions about the loan portfolio.")
    
    question = st.text_input("Example: 'How many people defaulted on their loan?'")
    if st.button("Query Database"):
        with st.spinner("Analyzing data..."):
            response = ask_data(question)
            if isinstance(response, dict):
                st.success("Business Insight:")
                st.write(response["insight"])
                
                with st.expander("View Technical Details (SQL & Raw Data)"):
                    st.code(response["sql"], language="sql")
                    st.write(response["raw_data"])
            else:
                st.error(response)