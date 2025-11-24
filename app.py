# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Student Final Score Predictor", layout="centered")
st.title("🎯 Student Final Score Predictor")

# --- 1) paths (adjust if needed) ---
MODEL_PATH = "model.pkl"  # the pipeline you saved in notebook
DATASET_PATH = "C:/Users/harin/Downloads/Task_students_performance_dataset.xlsx"  # used only to populate selectboxes

# --- 2) load model ---
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# --- 3) load dataset to get category options (optional but useful) ---
@st.cache_data
def load_df(path):
    return pd.read_excel(path)

try:
    df_ref = load_df(DATASET_PATH)
except Exception:
    df_ref = None

# --- 4) define feature lists exactly as in training ---
num_cols = [
    'Study_Hours_per_Week',
    'Attendance_Percentage',
    'Previous_Sem_Score',
    'Sleep_Hours',
    'Travel_Time',
    'Test_Anxiety_Level',
    'Motivation_Level',
    'Library_Usage_per_Week'
]
cat_cols = [
    'Gender',
    'Parental_Education',
    'Internet_Access',
    'Family_Income',
    'Tutoring_Classes',
    'Sports_Activity',
    'Extra_Curricular',
    'School_Type',
    'Peer_Influence',
    'Teacher_Feedback'
]

st.markdown("### Enter student details")

# --- 5) numeric inputs ---
numeric_inputs = {}
for col in num_cols:
    # reasonable default: mean from dataset if available, else 0
    default = float(df_ref[col].mean()) if (df_ref is not None and col in df_ref.columns and pd.api.types.is_numeric_dtype(df_ref[col])) else 0.0
    # pick a sensible min/max for UI convenience (you can tweak)
    numeric_inputs[col] = st.number_input(col.replace("_", " "), value=round(default, 2), format="%.2f")

# --- 6) categorical inputs (use dataset unique values if possible) ---
cat_inputs = {}
for col in cat_cols:
    if df_ref is not None and col in df_ref.columns:
        options = df_ref[col].dropna().unique().tolist()
        # sort strings nicely
        try:
            options = sorted(options)
        except Exception:
            pass
        # fallback: ask free text if single unique value only
        if len(options) > 0:
            cat_inputs[col] = st.selectbox(col.replace("_", " "), options)
        else:
            cat_inputs[col] = st.text_input(col.replace("_", " "), value="")
    else:
        # if no dataset loaded, accept free text
        cat_inputs[col] = st.text_input(col.replace("_", " "), value="")

# --- 7) assemble input DataFrame with exact column names and dtypes ---
input_dict = {}
input_dict.update({col: [numeric_inputs[col]] for col in num_cols})
input_dict.update({col: [cat_inputs[col]] for col in cat_cols})

input_df = pd.DataFrame(input_dict)

st.markdown("#### Input preview")
st.dataframe(input_df.T, use_container_width=True)

# --- 8) predict when button pressed ---
if st.button("Predict Final Score"):
    try:
        pred = model.predict(input_df)  # pipeline will preprocess
        st.success(f"Predicted Final Score: {pred[0]:.2f}")
    except Exception as e:
        st.error("Prediction failed. See error below.")
        st.exception(e)

st.markdown("---")

