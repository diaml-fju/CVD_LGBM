import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import re

st.sidebar.title("Important varialbes input")
#tab1, tab2, tab3, tab4 = st.tabs(["EOMG", "LOMG", "Thymoma", "Non-Thymoma"])
import re



# ------------------------- 共用函數：預測 + SHAP -------------------------
def predict_and_explain(model, x_train, input_df, model_name):
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import streamlit as st
    import re
    import xgboost as xgb

    st.subheader("Predict of Outcomes")

    try:
        # --- 特徵對齊 ---
        model_feature_names = model.get_booster().feature_names
        input_df = input_df[model_feature_names]
        background = x_train[model_feature_names]

        # --- 預測 ---
        proba = model.predict_proba(input_df)[0, 1]
        adaptive_thresholds = {
            "model A": 0.14298505,

        }
        threshold = adaptive_thresholds.get(model_name, 0.5)

        if proba >= threshold:
            st.error(f"Positive risk of ICU admission (probability={proba:.3f})")
        else:
            st.success(f"Negative risk of ICU admission (probability={proba:.3f})")

        # === KernelExplainer ===
        st.subheader("SHAP based personalized explanation")

        # ⚠️ 避免背景太大導致運算過久，只取樣本部分
        background_sample = background.sample(n=min(50, len(background)), random_state=42)

        # ⚠️ KernelExplainer 接受 predict_proba 的第二欄作為目標
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background_sample)

        # 只針對單筆輸入解釋
        input_row = input_df.iloc[[0]]
        shap_values = explainer.shap_values(input_row)

        # === SHAP 視覺化 ===
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_row.values[0],
                feature_names=input_row.columns.tolist()
            ),
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        import traceback
        st.error("⚠️ 發生錯誤，完整訊息如下：")
        st.text("".join(traceback.format_exception(None, e, e.__traceback__)))




# ✅ 定義通用二元選單函式
def binary_radio(label,key= None,index=None):
    return st.radio(
        label,
        options=[1, 0],
        format_func=lambda x: f"Yes" if x == 1 else f"No",
        index=index,
        key=key
    )

def binary_radio_Thymic(label,key= None,index=None):
    return st.radio(
        label,
        options=[1, 0],
        format_func=lambda x: f"Absence" if x == 0 else f"Presence",
        key=key,
        index=index
    )



# ------------------------- 模型 A -------------------------
def run_model_a_page():
    st.title("Personalized ICU Risk Prediction")
    st.markdown("""We 
provide detailed guidance through 
step-by-step instructions. Users can 
download the file below:""")
    with open("User guide for PredMGICU.pdf", "rb") as f:
        st.download_button(
            label="📥 Download of user-guide",
            data=f,
            file_name="User guide for PredMGICU.pdf",
            mime="application/pdf"
        )

    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_EOMG_2.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup2_EOMG.csv")
    x_train = x.drop(columns=[ "Y","MGFA clinical classification"])
    # 輸入變數
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        #Age = st.number_input("Age at onset (year)", 50,disabled=True)
        Gender = st.radio(
        "Gender",
        options=[(1, "Male"), (2, "Female")],
        format_func=lambda x: x[1],
        key="EOMG_Gender",
        index=1
        )
        Gender = Gender[0]  
        Disease_duration= st.number_input("Disease duration (month)", min_value=0.01, value=48.0, key="EOMG_Disease_duration")
        BMI = st.number_input("BMI", min_value=0.01, value=17.89)

    #MGFA
    # ➤ Corticosteroid variables
    with st.sidebar.expander("Treatment related variables", expanded=False):
    
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.0, value=10.0)
        Immunosuppressant = st.radio(
        "Immunosuppressant at admission", 
        options=[(1, "Azathioprine"), (2, "Calcineurin"), (3, "Mycophenolate"), (4, "Quinine"),(0, "None of above")], 
        format_func=lambda x: x[1],
        key="EOMG_Immuno",index = 4
        )
        Immunosuppressant = Immunosuppressant[0]
    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):
    
        Thymoma = binary_radio_Thymic("Thymoma", key="EOMG_Thymoma",index=0)
        Thymic = binary_radio_Thymic("Thymic hyperplasia", key="EOMG_Thymic",index=1)
        Thymectomy = binary_radio("Thymectomy", key="EOMG_Thymectomy",index=0)
    
    # ➤ Serology
    with st.sidebar.expander("Serology of autoantibody", expanded=False):

        Anti_AChR = binary_radio("Anti-AChR", key="EOMG_Anti_AChR",index=0)
        Anti_MuSK = binary_radio("Anti-MuSK", key="EOMG_Anti_MuSK",index=1)
        dSN = binary_radio("dSN", key="EOMG_dSN",index=1)

    # ➤ Comorbidity
    with st.sidebar.expander("Comorbidity variables", expanded=False):
        Infection = binary_radio("Infection at admission", key="EOMG_Infection",index=0)
        Thyroid = binary_radio("Thyroid disease", key="EOMG_Thyroid",index=1)
        Diabetes = binary_radio("Diabetes", key="EOMG_Diabetes",index=1)
        Hypertension = binary_radio("Hypertension", key="EOMG_Hypertension",index=1)
        Auto = binary_radio("Autoimmune disease", key="EOMG_Auto",index=1)
        ASCVD = binary_radio("ASCVD", key="EOMG_ASCVD",index=0)
        Chronic = binary_radio("Chronic lung disease", key="EOMG_Chronic",index=0)
        Good = binary_radio("Good syndrome", key="EOMG_Good",index=1)

    # ➤ Inflammation
    with st.sidebar.expander("Systemic inflammation markers variables", expanded=False):
        NLR = st.number_input("NLR", min_value=0.01, key="EOMG_NLR",value=4.286549708)
        PLR = st.number_input("PLR", min_value=0.01, key="EOMG_PLR",value=237.6115728)
        LMR = st.number_input("LMR", min_value=0.01, key="EOMG_LMR",value=2.23880597)
        SII = st.number_input("SII", min_value=0.01, key="EOMG_SII",value=1654608.2)
    
    # 建立 dict（易於維護）
    input_dict = {
    "Gender": Gender,
    "BMI": BMI,
    "Infection at admission": Infection,
    "Thyroid disease": Thyroid,
    "Autoimmune disease": Auto, 
    "Diabetes": Diabetes,
    "Hypertension": Hypertension,
    "ASCVD": ASCVD,
    "Chronic lung disease": Chronic,
    "Good syndrome": Good,
    "Disease duration (month)": Disease_duration,
    "Prednisolone daily dose before admission": Prednisolone,
    "Immunosuppressant at admission": Immunosuppressant,
    "Anti-MuSK": Anti_MuSK,
    "Anti-AChR": Anti_AChR,
    "dSN": dSN,
    "Thymoma": Thymoma,
    "Thymic hyperplasia": Thymic,
    "Thymectomy": Thymectomy,
    "NLR": NLR,
    "PLR": PLR,
    "LMR": LMR,
    "SII": SII
}

    


    if st.sidebar.button("Analysis"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        input_df = input_df.astype(float)
        predict_and_explain(model, x_train, input_df, "model A")

if model_choice == "EOMG":
    run_model_a_page()