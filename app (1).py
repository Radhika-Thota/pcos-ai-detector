import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="PCOS AI Detection",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("🔍 PCOS AI Detection System")
st.markdown("""
### Machine Learning Model for Polycystic Ovary Syndrome Risk Assessment
This AI tool analyzes clinical symptoms and biomarkers to assess PCOS risk using a trained Random Forest model.
""")

# Load ML model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load('pcos_model.joblib')
        scaler = joblib.load('scaler.joblib')
        st.success("✅ AI Model loaded successfully!")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        # Create a simple demo model as fallback
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scaler = StandardScaler()
        # Train on dummy data
        X_dummy = np.random.randn(100, 14)
        y_dummy = np.random.randint(0, 2, 100)
        scaler.fit(X_dummy)
        model.fit(scaler.transform(X_dummy), y_dummy)
        st.info("🔄 Using demo model for this session")
        return model, scaler

# Load the model
model, scaler = load_model()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Patient Information")

    with st.form("pcos_form"):
        # Basic Information
        st.subheader("👤 Basic Information")
        age = st.number_input("Age (years)", 10, 60, 25)
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 65.0, step=0.1)
        height = st.number_input("Height (cm)", 140.0, 200.0, 165.0, step=0.1)
        bmi = weight / ((height/100) ** 2)
        st.metric("Body Mass Index (BMI)", f"{bmi:.1f}")

        # Menstrual History
        st.subheader("🩸 Menstrual History")
        cycle_regularity = st.radio(
            "Menstrual Cycle Regularity",
            ["Regular (21-35 days between periods)", "Irregular (more than 35 days between periods)"]
        )
        cycle_length = st.number_input("Typical Cycle Length (days)")

        # Clinical Symptoms
        st.subheader("🔍 Clinical Symptoms")

# The list of options that will appear in the dropdown
        # 1. Define the mapping dictionary
        acne_mapping = {
            "None": 0,
            "Mild": 1,
            "Moderate": 2,
            "Severe": 3
        }

# 2. Get the string value from the selectbox
        acne = st.selectbox(
           "Acne Severity",
            options=["None", "Mild", "Moderate", "Severe"],
            index=0  # Default to "Mild"
           
        )
        acne_score = acne_mapping.get(acne, 0)
        hair_growth_mapping = {
            "None": 0,
            "Mild": 1,
            "Moderate": 2,
            "Severe": 3
        }
        hair_growth = st.selectbox(
            "Hair Growth",
            options=["None", "Mild", "Moderate", "Severe"],
            index=0  # Default to "Mild"
        )
        hair_growth_score = hair_growth_mapping.get(hair_growth, 0)

        hair_loss = st.radio("Hair Thinning or Loss", ["No", "Yes"])
        skin_darkening = st.radio("Dark Skin Patches (neck/armpits)", ["No", "Yes"])

        # Laboratory Values
        st.subheader("🧪 Laboratory Values")
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", 0, 500)
        fasting_insulin = st.number_input("Fasting Insulin (μIU/mL)", 0, 100)
        lh_level = st.number_input("LH Level (mIU/mL)", 0, 50)
        fsh_level = st.number_input("FSH Level (mIU/mL)", 0, 55)

        # Submit button
        submitted = st.form_submit_button("🔍 Assess PCOS Risk")

with col2:
    st.header("📊 Assessment Results")

    if submitted:
        # Calculate derived features
        lh_fsh_ratio = lh_level / fsh_level if fsh_level > 0 else 0
        insulin_resistance = (fasting_glucose * fasting_insulin) / 405

        # Prepare feature array for ML model
        features = np.array([[
            age,                                    # age
            bmi,                                    # bmi
            0 if "Irregular" in cycle_regularity else 1,  # cycle_regularity
            cycle_length,                           # cycle_length_days
            acne_score,                                   # acne_severity
            hair_growth_score,                            # hair_growth
            1 if hair_loss == "Yes" else 0,         # hair_loss
            1 if skin_darkening == "Yes" else 0,    # skin_darkening
            fasting_glucose,                        # fasting_glucose
            fasting_insulin,                        # fasting_insulin
            lh_level,                               # lh_level
            fsh_level,                              # fsh_level
            lh_fsh_ratio,                           # lh_fsh_ratio
            insulin_resistance                      # insulin_resistance
        ]])

        try:
            # Scale features and make prediction
            features_scaled = scaler.transform(features)
            probability = model.predict_proba(features_scaled)[0][1] * 100

            # Display risk assessment
            if probability >= 70:
                risk_color = "#ff4444"
                risk_level = "HIGH RISK"
                recommendation = "We strongly recommend consulting with an endocrinologist or gynecologist for comprehensive evaluation and personalized treatment planning."
            elif probability >= 30:
                risk_color = "#ffaa00"
                risk_level = "MODERATE RISK"
                recommendation = "Consider discussing these symptoms with your healthcare provider. Lifestyle modifications and regular monitoring may be beneficial."
            else:
                risk_color = "#00c851"
                risk_level = "LOW RISK"
                recommendation = "Your current profile suggests low risk. Maintain healthy lifestyle habits and schedule regular wellness check-ups."

            # Results display
            st.markdown(f"""
            <div style='
                padding: 25px;
                border-radius: 10px;
                border-left: 5px solid {risk_color};
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                margin: 20px 0;
            '>
                <h2 style='color: {risk_color}; margin: 0 0 10px 0;'>{risk_level}</h2>
                <h1 style='color: {risk_color}; font-size: 3.5em; margin: 10px 0;'>{probability:.1f}%</h1>
                <p style='font-size: 1.1em; line-height: 1.6;'><strong>Recommendation:</strong> {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

            # Feature importance visualization
            if hasattr(model, 'feature_importances_'):
                st.subheader("🔍 Key Contributing Factors")

                feature_names = [
                    'Age', 'BMI', 'Cycle Regularity', 'Cycle Length',
                    'Acne', 'Hair Growth', 'Hair Loss', 'Skin Darkening',
                    'Glucose', 'Insulin', 'LH Level', 'FSH Level',
                    'LH/FSH Ratio', 'Insulin Resistance'
                ]

                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(8)

                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top Features Influencing This Prediction",
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Additional insights
            st.subheader("💡 Clinical Insights")
            insights = []

            if bmi > 25:
                insights.append("• **Weight management** may help improve metabolic parameters")
            if "Irregular" in cycle_regularity:
                insights.append("• **Cycle tracking** can provide valuable information for your healthcare provider")
            if insulin_resistance > 2.5:
                insights.append("• **Dietary modifications** focusing on low glycemic index foods may be beneficial")
            if lh_fsh_ratio > 2.5:
                insights.append("• **Hormonal evaluation** may provide additional diagnostic insights")

            for insight in insights:
                st.write(insight)

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check that all input values are within reasonable ranges.")

    else:
        # Placeholder before form submission
        st.info("""
        👈 **Please fill out the patient information form to get started**

        The AI model will analyze:
        - Demographic information
        - Menstrual history
        - Clinical symptoms
        - Laboratory values
        - Metabolic parameters

        **Click 'Assess PCOS Risk' to see your personalized assessment.**
        """)

# Footer with disclaimer
st.markdown("---")
st.markdown("""
<div style='background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;'>
    <h4 style='color: #856404; margin: 0 0 10px 0;'>📝 Medical Disclaimer</h4>
    <p style='color: #856404; margin: 0;'>
        This AI tool is for educational and screening purposes only. It is not a substitute for
        professional medical advice, diagnosis, or treatment. Always consult qualified healthcare
        providers for medical concerns. The predictions are based on machine learning patterns
        and should be interpreted in clinical context by healthcare professionals.
    </p>
</div>
""", unsafe_allow_html=True)

# Model information
with st.sidebar:
    st.header("🤖 Model Information")
    st.markdown("""
    **Algorithm:** Random Forest Classifier
    **Features:** 14 clinical parameters
    **Training:** Synthetic dataset with realistic patterns
    **Purpose:** Early risk assessment and screening

    ---
    **Developed for:** Apollo Innovatus Hackathon 2025
    **Theme:** Advancing Women's Health
    **Team:** PCOS Sentinel
    """)
