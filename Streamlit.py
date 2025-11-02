import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

st.set_option("deprecation.showPyplotGlobalUse", False)

# ================== Sidebar ==================
st.sidebar.title("Model / page")
page = st.sidebar.selectbox("", ["CVD demo"])   # 之後要多頁再加


# ================== 小工具：拿特徵名 ==================
def get_feature_names_from_model_or_data(model, fallback_cols):
    """
    HistGradientBoostingClassifier 沒有 get_booster()
    但有 feature_names_in_（只要是用 DataFrame / 有欄位名 fit 的話）
    這邊先嘗試從 model 拿，拿不到就退回訓練檔的欄位
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_cols)


# ================== 共用：預測 + SHAP ==================
def predict_and_explain(model, x_train, input_df, model_name="HGB"):
    st.subheader("Prediction")

    # 1) 對齊欄位
    model_feature_names = get_feature_names_from_model_or_data(model, x_train.columns)
    input_df = input_df[model_feature_names]
    background = x_train[model_feature_names]

    # 2) 預測機率（HGB 是分類的話一樣有 predict_proba）
    proba = model.predict_proba(input_df)[0, 1]
    st.write(f"🔢 Predicted probability: **{proba:.3f}**")

    # 3) 自適應門檻（你可以自己改數字 / 換字典）
    adaptive_thresholds = {
        "HGB": 0.14298505,
    }
    threshold = adaptive_thresholds.get(model_name, 0.5)

    if proba >= threshold:
        st.error(f"Predicted: **Positive** (prob >= {threshold:.3f})")
    else:
        st.success(f"Predicted: **Negative** (prob < {threshold:.3f})")

    # 4) SHAP 解釋
    st.subheader("SHAP explanation")

    # 背景不要太多，不然 KernelExplainer 會很慢
    background_sample = background.sample(
        n=min(50, len(background)),
        random_state=42
    )

    # ★ 重點：
    # SHAP 的 TreeExplainer 對 XGBoost / LightGBM / sklearn 的 Tree / RF 都很 ok
    # 但對 HistGradientBoosting 有時候會直接不支援
    # 所以這裡「先試」，失敗就退回 KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)

        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    except Exception:
        # 退回通用版本
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


# ================== 頁面：你的這個專案 ==================
def run_cvd_demo_page():
    st.title("CVD prediction (HistGradientBoostingClassifier)")

    # 1. 讀模型（重點：sklearn 的要用 pickle / joblib 讀，不是 load_model）
    with open(r"CVD_HBB.joblib", "rb") as f:   # ← ← ← ① 這裡換成你的模型路徑
        model = pickle.load(f)

    # 2. 讀訓練資料，單純是為了拿欄位結構
    x = pd.read_csv(r"CVD_SHAP_Model.csv")     # ← ← ← ② 這裡換成你的訓練資料
    x_train = x.drop(columns=["Y"])            # ← ← ← ③ 如果你的 label 不是叫 Y，要改

    # 3. Streamlit 輸入欄位（這裡先放你剛剛那幾個）
    st.write("### Input variables")
    NIHSS = st.number_input("NIHSS", min_value=0.0, value=1.0, step=0.1)
    HR_Max = st.number_input("HR_Max", min_value=0.0, value=85.0, step=0.1)
    BT_Mean = st.number_input("BT_Mean", min_value=0.0, value=36.2875, step=0.001)
    SBP_Mean = st.number_input("SBP_Mean", min_value=0.0, value=156.416667, step=0.1)
    BT_std = st.number_input("BT_std", min_value=0.0, value=0.309989919, step=0.001)

    # 把用戶輸入先丟進 dict
    user_inputs = {
        "NIHSS": NIHSS,
        "HR_Max": HR_Max,
        "BT_Mean": BT_Mean,
        "SBP_Mean": SBP_Mean,
        "BT_std": BT_std,
    }

    # 4. 按鈕觸發
    if st.sidebar.button("Analysis"):
        # 依照訓練資料欄位順序組一筆資料
        row = []
        for col in x_train.columns:
            row.append(user_inputs.get(col, 0.0))   # 沒畫在畫面的欄位先補 0
        input_df = pd.DataFrame([row], columns=x_train.columns)

        # 保險起見轉成 float
        input_df = input_df.astype(float)

        # 丟去跑預測 + SHAP
        predict_and_explain(model, x_train, input_df, "HGB")


# ================== 主流程 ==================
if page == "CVD demo":
    run_cvd_demo_page()
