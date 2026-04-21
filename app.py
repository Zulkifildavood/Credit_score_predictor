import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- 1. AUTHENTICATION (HARDCODED FOR NOW) ---
# Room for development: Later, replace 'check_password' with a database or AWS Cognito lookup
def check_password():
    def password_entered():
        if st.session_state["username"] == "admin" and st.session_state["password"] == "tcs2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True

if check_password():
    # --- 2. LOAD BRAIN (MODEL & SCALER) ---
    @st.cache_resource
    def load_assets():
        with open('credit_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    model, scaler = load_assets()

    # --- 3. UI LAYOUT ---
    st.title("🏦 TCS Credit Scoring System")
    st.markdown("Automated loan risk assessment platform.")

    menu = ["Single Prediction", "Batch Analysis", "Model Insights"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Single Prediction":
        st.subheader("New Loan Application")
        
        with st.form("loan_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, step=1)
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                income_annum = st.number_input("Annual Income (INR)", min_value=0)
                loan_amount = st.number_input("Loan Amount (INR)", min_value=0)
            
            with col2:
                loan_term = st.number_input("Loan Term (Years)", min_value=1)
                cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
                residential_assets = st.number_input("Residential Assets Value (INR)", min_value=0)
                commercial_assets = st.number_input("Commercial Assets Value (INR)", min_value=0)
                luxury_assets = st.number_input("Luxury Assets Value (INR)", min_value=0)
                bank_assets = st.number_input("Bank Assets Value (INR)", min_value=0)
            
            submit = st.form_submit_button("Predict Loan Status")

        if submit:
            # --- FEATURE ENGINEERING LAYER ---
            # Mapping strings to numbers exactly as your training script did
            data = {
                'no_of_dependents': no_of_dependents,
                'education': 1 if education == "Graduate" else 0,
                'self_employed': 1 if self_employed == "Yes" else 0,
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'residential_assets_value': residential_assets,
                'commercial_assets_value': commercial_assets,
                'luxury_assets_value': luxury_assets,
                'bank_asset_value': bank_assets
            }
            
            input_df = pd.DataFrame([data])
            scaled_input = scaler.transform(input_df)
            
            # --- PREDICTION ---
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            # --- VISUALIZATION (CREDIT GAUGE) ---
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Approval Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig)

            if prediction == 1:
                st.success(f"✅ Approved! Probability Score: {probability:.2f}")
            else:
                st.error(f"❌ Rejected. Probability Score: {probability:.2f}")

    elif choice == "Model Insights":
        st.subheader("Decision Drivers (Feature Importance)")
        # Calculate importance from coefficients
        importance = model.coef_[0]
        feature_names = ['Dependents', 'Education', 'Self-Employed', 'Income', 'Loan Amt', 'Term', 'CIBIL', 'Res. Assets', 'Comm. Assets', 'Lux. Assets', 'Bank Assets']
        
        feat_df = pd.DataFrame({'Feature': feature_names, 'Impact': importance})
        feat_df = feat_df.sort_values(by='Impact', ascending=False)

        fig_importance = px.bar(feat_df, x='Impact', y='Feature', orientation='h', 
                                 title="What influences the Decision Most?",
                                 color='Impact', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_importance)
        st.write("Positive values push toward Approval; negative values push toward Rejection.")