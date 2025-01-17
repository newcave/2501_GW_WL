import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 📂 파일 업로드 및 데이터 로딩
@st.cache_data
def load_data(file):
    data = pd.read_excel(file, header=2)
    data.columns = data.columns.str.strip()
    return data

# 📄 기본 파일 로딩 함수
@st.cache_data
def load_default_data():
    data = pd.read_excel("GW_001.xlsx", header=2)
    data.columns = data.columns.str.strip()
    return data

# ⚙️ 사이드바 설정
st.sidebar.title("🛰️ 지하수위 예측 설정")

# 📁 지점 선택 체크박스
use_default = st.sidebar.checkbox("✅ 기본 데이터(GW_001.xlsx) 사용")
uploaded_file = st.sidebar.file_uploader("📤 지점 데이터 업로드 (Excel 파일)", type=["xlsx"])

# 📅 리드 타임 설정
lead_time = st.sidebar.slider("⏳ 리드 타임 (예측 기간, 일)", min_value=1, max_value=30, value=7)

# 🔄 룩백 설정
look_back = st.sidebar.slider("🔍 룩백 기간 (과거 데이터 사용 기간, 일)", min_value=1, max_value=365, value=30)

# 🔧 하이퍼파라미터 설정
n_estimators = st.sidebar.slider("🛠️ # of Estimators (하이퍼파라미터)", min_value=10, max_value=500, step=10, value=100)

# 📊 데이터 로딩 및 출력
if uploaded_file or use_default:
    if uploaded_file:
        data = load_data(uploaded_file)
    else:
        data = load_default_data()
    
    # '계측수위' 컬럼 고정 사용
    wl_column = '계측수위'
    if wl_column in data.columns:
        data = data.sort_values(wl_column).reset_index(drop=True)
        st.success("✅ raw data 처리 및 모델 실행 ✅")
    else:
        st.error("❌ 데이터에 '계측수위' 컬럼이 없습니다. 업로드한 데이터를 확인하세요.")
        st.stop()

    with st.expander("🔍 Raw 데이터 보기", expanded=True):
        st.write(data)
        st.write(f"📋 **데이터 컬럼명:** {list(data.columns)}")
    
    # 📌 독립변수 선택
    st.subheader("📈 독립변수 선택")
    independent_vars = st.multiselect("✅ 사용할 독립변수 선택:", options=list(data.columns), default=[col for col in ["수온", "전도도", wl_column] if col in data.columns])

    # 🎯 예측변수 선택
    st.subheader("🎯 예측변수 선택")
    target_var = st.selectbox("✅ 예측할 변수 선택:", options=list(data.columns), index=list(data.columns).index(wl_column))

    # 🔒 변수 설정 완료 버튼 추가
    if st.button("🚀 변수 설정 완료"):
        with st.expander("📌 선택한 변수 보기", expanded=False):
            st.write(f"✅ **선택한 독립변수:** {independent_vars}")
            st.write(f"🎯 **예측 변수:** {target_var}")
            st.write(f"⏳ **리드 타임:** {lead_time}일, 🔍 **룩백 기간:** {look_back}일, 🛠️ **Estimator 수:** {n_estimators}")
        
        # 📊 EDA 시각화
        st.subheader("🔎 기본 EDA")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, var in enumerate(independent_vars[:3]):
            sns.histplot(data[var], kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f'{var} Distribution')
        st.pyplot(fig)
        
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(data[independent_vars + [target_var]].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title('Feature Correlation Heatmap')
        st.pyplot(fig_corr)

    # 🤖 모델 학습 및 예측
    if st.button("📊 모델 실행"):
        X = data[independent_vars].apply(pd.to_numeric, errors='coerce').dropna()
        y = pd.to_numeric(data[target_var], errors='coerce').loc[X.index].dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)

        # 📊 결과 시각화
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        axes[0].plot(y_train.values, label='실제값', linewidth=2)
        axes[0].plot(y_train_pred, label='예측값', linestyle='--', linewidth=2)
        axes[0].set_title(f'Training Set\nRMSE: {train_rmse:.2f}, R2: {train_r2:.2f}')
        axes[0].legend()

        axes[1].plot(y_test.values, label='실제값', linewidth=2)
        axes[1].plot(y_pred, label='예측값', linestyle='--', linewidth=2)
        axes[1].set_title(f'Testing Set\nRMSE: {test_rmse:.2f}, R2: {test_r2:.2f}')
        axes[1].legend()

        axes[2].scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
        axes[2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        axes[2].set_title(f'Actual vs Predicted\nR2: {test_r2:.2f}')
        axes[2].set_xlabel('Actual')
        axes[2].set_ylabel('Predicted')

        st.pyplot(fig)
else:
    st.info("💡 **K-water AI LAB x Groundwater Research Team Collaboration.**")
    col1, col2 = st.columns(2)
    with col1:
        st.image("FIG2.png", caption="📍 위치도 및 염분 분포도")
    with col2:
        st.video("media.mp4")
