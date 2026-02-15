import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Diabetes Care Assistant",
    page_icon="🩺",
    layout="wide"
)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Highlighted Title Section
# -------------------------

st.markdown("""
    <div style='
        text-align: center;
        background-color: #D6EAF8;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.2);
    '>
        <h1 style='color:#1B4F72; font-size:45px; margin-bottom:10px;'>
            🩺 AI Diabetes Care Assistant
        </h1>
        <p style='color:#154360; font-size:18px;'>
            Machine Learning Based Diabetes Risk Prediction & Health Guidance
        </p>
    </div>
""", unsafe_allow_html=True)

st.warning("⚠ This system is not a replacement for professional medical advice.")

st.markdown("---")

# -------------------------
# Diabetes Prediction Section
# -------------------------

st.subheader("📊 Diabetes Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, key="preg")
    glucose = st.number_input("Glucose Level", min_value=0, key="glucose")
    bp = st.number_input("Blood Pressure", min_value=0, key="bp")
    skin = st.number_input("Skin Thickness", min_value=0, key="skin")

with col2:
    insulin = st.number_input("Insulin", min_value=0, key="insulin")
    bmi = st.number_input("BMI", min_value=0.0, key="bmi")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, key="dpf")
    age = st.number_input("Age", min_value=0, key="age")

if st.button("🔍 Predict Risk"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("🚨 High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.info(f"📈 Probability of Diabetes: {round(probability[0][1]*100,2)}%")

st.markdown("---")

# -------------------------
# Chatbot Section
# -------------------------

st.subheader("💬 Health Recommendation Chatbot")

user_input = st.text_input("Ask about diet, exercise, symptoms, precautions", key="chat_input")

def chatbot_response(text):
    text = text.lower()

    if "diet" in text:
        return "🥗 Eat low sugar food, avoid sugary drinks, increase fiber intake."
    elif "exercise" in text:
        return "🏃 Walk at least 30 minutes daily and maintain physical activity."
    elif "symptom" in text:
        return "⚠ Common symptoms include frequent urination, fatigue, and increased thirst."
    elif "precaution" in text:
        return "🛡 Monitor blood sugar regularly and maintain healthy lifestyle."
    else:
        return "Please ask about diet, exercise, symptoms or precautions."

if user_input:
    response = chatbot_response(user_input)
    st.success(response)
