import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier

st.set_page_config(page_title="PD Macro Analytics", layout="wide", page_icon="🏦")

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_final_model.cbm"))
    return model

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DIR / "test_fe.csv")
    X_static = df.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore').sample(2000, random_state=42)
    
    cat_cols = X_static.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X_static[col] = X_static[col].fillna('Missing')
        
    return X_static

def predict_pd(model, X_static, repo_rate, vix, unemployment, px, czk):
    X_scen = X_static.copy()
    X_scen['repo_rate'] = repo_rate
    X_scen['vix'] = vix
    X_scen['unemployment'] = unemployment
    X_scen['px'] = px
    X_scen['czk_eur'] = czk
    return model.predict_proba(X_scen)[:, 1].mean()

# --- ИНТЕРФЕЙС ---
st.title("🏦 Система прогнозирования кредитного риска (PD)")
st.markdown("Сценарный анализ вероятности дефолта портфеля заемщиков с учетом макроэкономических факторов Чехии.")

with st.spinner("Загрузка модели и данных..."):
    model = load_model()
    X_static = load_data()

st.sidebar.header("🎛️ Макроэкономические параметры")
repo_rate = st.sidebar.slider("Ставка ЦБ (%)", 0.0, 15.0, 4.0, 0.25)
vix = st.sidebar.slider("Индекс VIX", 10.0, 80.0, 18.0, 1.0)
unemployment = st.sidebar.slider("Безработица (%)", 1.0, 10.0, 3.0, 0.1)
px = st.sidebar.slider("Индекс PX", 500.0, 2000.0, 1400.0, 10.0)
czk_eur = st.sidebar.slider("Курс CZK/EUR", 20.0, 30.0, 24.5, 0.1)

# Считаем текущий PD
current_pd = predict_pd(model, X_static, repo_rate, vix, unemployment, px, czk_eur)

col1, col2 = st.columns([1, 1])

with col1:
    st.metric("Средний PD портфеля", f"{current_pd * 100:.2f}%")
    st.markdown("""
    **Как это работает:** 
    Вы меняете макро-факторы слева. Обученная модель CatBoost мгновенно пересчитывает вероятность дефолта для портфеля реальных заемщиков.
    """)

with col2:
    # Считаем фиксированные сценарии
    pd_opt = predict_pd(model, X_static, 2.0, 12.0, 2.0, 1800, 24.0)
    pd_base = predict_pd(model, X_static, 4.0, 18.0, 3.2, 1500, 24.5)
    pd_crisis = predict_pd(model, X_static, 8.0, 60.0, 7.0, 800, 28.0)
    
    # Создаем DataFrame для графика
    chart_data = pd.DataFrame({
        "Сценарий": ["Оптимистичный", "Базовый", "Кризисный"],
        "PD (%)": [pd_opt*100, pd_base*100, pd_crisis*100]
    }).set_index("Сценарий")
    
    st.write("**Сравнение сценариев:**")
    st.bar_chart(chart_data, color="#FF4B4B")

# Ожидаемые потери
st.divider()
st.subheader("Ожидаемые потери (Expected Loss)")

portfolio_size = 100_000_000 # 100 млн
lgd = 0.45 # Loss Given Default

el = current_pd * lgd * portfolio_size

st.write(f"При портфеле в 100 млн и текущих макро-параметрах, ожидаемые потери составят: **{el:,.2f}**")