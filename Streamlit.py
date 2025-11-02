import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from joblib import load

# 不要這行，因為你的版本不認得
# st.set_option("deprecation.showPyplotGlobalUse", False)

st.sidebar.title("Model / page")
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
    background = x_train[model_feature_names]

    proba = model.predict_proba(input_df)[0, 1]
    st.write(f"🔢 Predicted probability: **{proba:.3f}**")

    adaptive_thresholds = {"HGB": 0.23826015749222382}
    threshold = adaptive_thresholds.get(model_name, 0.5)

    if proba >= threshold:
        st.error(f"Predicted: **Positive** (prob ≥ {threshold:.3f})")
    else:
        st.success(f"Predicted: **Negative** (prob < {threshold:.3f})")

    st.subheader("SHAP explanation")

    background_distribution=x_train

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)

        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    except Exception:
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            background_distribution
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
    st.title("CVD prediction (HistGradientBoostingClassifier)")

    # ① 換成你的模型
    with open(r"CVD_HGB.joblib", "rb") as f:
        model = load(f)

    # ② 換成你的訓練資料
    x = pd.read_csv(r"CVD_SHAP_Model.csv")
    # ③ 如果你的 y 不是叫 Y，就改這裡
    x_train = x.drop(columns=["Y"])

    st.write("### Input variables")
    NIHSS = st.sidebar.number_input("NIHSS", min_value=0.0, value=1.0, step=0.1)
    HR_Max = st.sidebar.number_input("HR_Max", min_value=0.0, value=85.0, step=0.1)
    BT_Mean = st.sidebar.number_input("BT_Mean", min_value=0.0, value=36.2875, step=0.001)
    SBP_Mean = st.sidebar.number_input("SBP_Mean", min_value=0.0, value=156.416667, step=0.1)
    BT_std = st.sidebar.number_input("BT_std", min_value=0.0, value=0.309989919, step=0.001)

    user_inputs = {
        "NIHSS": NIHSS,
        "HR_Max": HR_Max,
        "BT_Mean": BT_Mean,
        "SBP_Mean": SBP_Mean,
        "BT_std": BT_std,
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
