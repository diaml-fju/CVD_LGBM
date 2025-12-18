import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from joblib import load

# 不要這行，因為你的版本不認得
# st.set_option("deprecation.showPyplotGlobalUse", False)

st.sidebar.title("Input Panel")
#page = st.sidebar.selectbox("", ["CVD demo"])
page = "CVD demo"

def get_feature_names_from_model_or_data(model, fallback_cols):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_cols)


def predict_and_explain(model, x_train, input_df, model_name="HGB"):
    st.subheader("Prediction")

    model_feature_names = get_feature_names_from_model_or_data(model, x_train.columns)
    input_df = input_df[model_feature_names]
    #background = x_train[model_feature_names]

    proba = model.predict_proba(input_df)[0, 1]
    

    adaptive_thresholds = {"HGB": 0.23826015749222382}
    threshold = adaptive_thresholds.get(model_name, 0.5)

    if proba >= threshold:
        st.error(f"Predicted: **Positive** ")
    else:
        st.success(f"Predicted: **Negative** ")

    st.subheader("SHAP explanation")

    background_sample=x_train
    #background = x_train[model_feature_names]
    #background_sample = background.sample(n=min(100, len(background)), random_state=42)
    try:
        explainer = shap.KernelExplainer(model.predict_proba,background_sample)
        shap_values = explainer.shap_values(x_train)

        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    except Exception:
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            background_sample
        )
        row = input_df.iloc[[0]]
        sv = explainer.shap_values(row)

        shap.plots.waterfall(
            shap.Explanation(
                values=sv[0],
                base_values=explainer.expected_value,
                data=row.values[0],
                feature_names=row.columns.tolist()
            ),
            show=False
        )
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)


def run_cvd_demo_page():
    st.title("Machine Learning Model to Predict In-hospital Stay of Ischemic Stroke Patients")

    # ① 換成你的模型
    with open(r"CVD_HGB.joblib", "rb") as f:
        model = load(f)

    # ② 換成你的訓練資料
    x = pd.read_csv(r"CVD_SHAP_Model.csv")
    # ③ 如果你的 y 不是叫 Y，就改這裡
    x_train = x.drop(columns=["Y"])

    
    NIHSS = st.sidebar.number_input("NIHSS", min_value=0,max_value=42, value=1, step=1)
    HR_Max = st.sidebar.number_input("HR_Max", min_value=0, value=108, step=1)
    BT_Mean = st.sidebar.number_input("BT_Mean", min_value=0.0, value=37.1, step=0.1,
    format="%.1f")
    BT_std = st.sidebar.number_input("BT_std", min_value=0.0, value=0.9, step=0.1,
    format="%.1f")
    SBP_Mean = st.sidebar.number_input("SBP_Mean", min_value=0.0, value=149.7, step=0.1,
    format="%.1f")
    user_inputs = {
        "NIHSS": NIHSS,
        "HR_Max": HR_Max,
        "BT_Mean": BT_Mean,
        "BT_std": BT_std,
        "SBP_Mean": SBP_Mean,
        
    }

    if st.sidebar.button("Analysis"):
        row = []
        for col in x_train.columns:
            row.append(user_inputs.get(col, 0.0))
        input_df = pd.DataFrame([row], columns=x_train.columns)
        input_df = input_df.astype(float)

        predict_and_explain(model, x_train, input_df, "HGB")


if page == "CVD demo":
    run_cvd_demo_page()
