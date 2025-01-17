import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 파일 업로드 및 데이터 로딩
@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

# 기본 파일 로딩 함수
@st.cache_data
def load_default_data():
    data = pd.read_excel("GW_001.xlsx")
    return data

# 사이드바 설정
st.sidebar.title("지하수위 예측 설정")

# 지점 선택 체크박스
use_default = st.sidebar.checkbox("기본 데이터(GW_001.xlsx) 사용")
uploaded_file = st.sidebar.file_uploader("지점 데이터 업로드 (Excel 파일)", type=["xlsx"])

# 리드 타임 설정
lead_time = st.sidebar.slider("리드 타임 (예측 기간, 일)", min_value=1, max_value=30, value=7)

# 룩백 설정
look_back = st.sidebar.slider("룩백 기간 (과거 데이터 사용 기간, 일)", min_value=1, max_value=365, value=30)

# 하이퍼파라미터 설정
n_estimators = st.sidebar.slider("# of Estimators (하이퍼파라미터)", min_value=10, max_value=500, step=10, value=100)

# 데이터 로딩 및 출력
if uploaded_file or use_default:
    if uploaded_file:
        data = load_data(uploaded_file)
    else:
        data = load_default_data()
    
    with st.expander("Raw 데이터 보기", expanded=False):
        st.write(data)
    
    # 독립변수 선택
    st.subheader("독립변수 선택")
    independent_vars = st.multiselect(
        "사용할 독립변수 선택:",
        options=["EC", "Temperature", "WL"],
        default=["EC", "Temperature", "WL"]
    )
    
    # 예측변수 선택
    st.subheader("예측변수 선택")
    target_var = st.selectbox(
        "예측할 변수 선택:",
        options=["WL"],
        index=0
    )
    
    # 데이터 및 설정 확인
    st.write(f"선택한 독립변수: {independent_vars}")
    st.write(f"예측 변수: {target_var}")
    st.write(f"리드 타임: {lead_time}일, 룩백 기간: {look_back}일, Estimator 수: {n_estimators}")
    
    # 모델 학습 및 예측
    if st.button("모델 실행"):
        X = data[independent_vars]
        y = data[target_var]
        
        # 결측치 제거
        X = X.dropna()
        y = y.loc[X.index]
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest 모델 학습
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측 및 성능 평가
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # 결과 출력
        st.subheader("모델 예측 결과")
        st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        st.line_chart(pd.DataFrame({"실제값": y_test.values, "예측값": y_pred}, index=y_test.index))
else:
    st.info("좌측 사이드바에서 데이터를 업로드하거나 기본 데이터를 선택하세요.")
