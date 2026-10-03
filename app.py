import streamlit as st
import pandas as pd
import numpy as np
import joblib
from inputs_transformer import InputsTransformer

model = joblib.load('Models/linear_regression.pkl')

st.title("🎓 Exam Score Predictor")
st.write("Enter the student's information to predict the exam score.")

with st.form("prediction_form"):
    Hours_Studied = st.number_input("Hours Studied", min_value=0, step=1)
    Attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)

    Parental_Involvement = st.selectbox(
        "Parental Involvement", ["Low", "Medium", "High"]
    )
    Access_to_Resources = st.selectbox(
        "Access to Resources", ["High", "Medium", "Low"]
    )
    Extracurricular_Activities = st.selectbox(
        "Extracurricular Activities", ["No", "Yes"]
    )

    Sleep_Hours = st.number_input("Sleep Hours", min_value=0, step=1)
    Previous_Scores = st.number_input("Previous Scores", min_value=0, step=1)

    Motivation_Level = st.selectbox(
        "Motivation Level", ["Low", "Medium", "High"]
    )
    Internet_Access = st.selectbox("Internet Access", ["Yes", "No"])

    Tutoring_Sessions = st.number_input(
        "Tutoring Sessions", min_value=0, step=1
    )

    Family_Income = st.selectbox(
        "Family Income", ["Low", "Medium", "High"]
    )
    Teacher_Quality = st.selectbox(
        "Teacher Quality", ["Medium", "High", "Low"]
    )
    School_Type = st.selectbox("School Type", ["Public", "Private"])
    Peer_Influence = st.selectbox(
        "Peer Influence", ["Positive", "Negative", "Neutral"]
    )

    Physical_Activity = st.number_input(
        "Physical Activity", min_value=0, step=1
    )

    Learning_Disabilities = st.selectbox(
        "Learning Disabilities", ["No", "Yes"]
    )
    Parental_Education_Level = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )
    Distance_from_Home = st.selectbox(
        "Distance from Home", ["Near", "Moderate", "Far"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("Predict Exam Score")

if submitted:
    input_data = pd.DataFrame([{
        "Hours_Studied": Hours_Studied,
        "Attendance": Attendance,
        "Parental_Involvement": Parental_Involvement,
        "Access_to_Resources": Access_to_Resources,
        "Extracurricular_Activities": Extracurricular_Activities,
        "Sleep_Hours": Sleep_Hours,
        "Previous_Scores": Previous_Scores,
        "Motivation_Level": Motivation_Level,
        "Internet_Access": Internet_Access,
        "Tutoring_Sessions": Tutoring_Sessions,
        "Family_Income": Family_Income,
        "Teacher_Quality": Teacher_Quality,
        "School_Type": School_Type,
        "Peer_Influence": Peer_Influence,
        "Physical_Activity": Physical_Activity,
        "Learning_Disabilities": Learning_Disabilities,
        "Parental_Education_Level": Parental_Education_Level,
        "Distance_from_Home": Distance_from_Home,
        "Gender": Gender
    }])

    transformer = InputsTransformer(input_data, '/home/mohamed/Documents/Github/Gho-byte/EduPredict/NoteBooks/fit.csv')
    transformed_input = transformer.get_result()
    prediction = model.predict(transformed_input)[0]
    prediction = transformer.inverse_transformation('Exam_Score', prediction)
    prediction = np.float16(prediction[0])

    st.success(f"📊 Predicted Exam Score: **{prediction}**")