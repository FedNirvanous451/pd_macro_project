import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

def get_portfolio_pd(model, X_static, macro_data):
    """
    Считает средний PD для портфеля клиентов при заданных макро-факторах.
    """
    X_scenario = X_static.copy()
    
    # 1. Подставляем макро-факторы (числа) и явно приводим к float
    for col in macro_data.columns:
        X_scenario[col] = float(macro_data[col].values[0])
        
    # 2. ПРИНУДИТЕЛЬНО приводим ВСЕ колонки, которые НЕ являются макро, к строке
    # (это спасет от ошибки с feature_idx)
    macro_cols = ['repo_rate', 'czk_eur', 'unemployment', 'px', 'vix']
    
    for col in X_scenario.columns:
        if col not in macro_cols:
            X_scenario[col] = X_scenario[col].fillna('Missing').astype(str)
            
    # 3. Теперь макро-факторы снова превращаем в float 
    # (потому что после шага 2 они стали строками, но мы их не трогали в цикле выше,
    # но на всякий случай убедимся)
    for col in macro_cols:
        if col in X_scenario.columns:
            X_scenario[col] = X_scenario[col].astype(float)

    # 4. Предсказание
    pd_proba = model.predict_proba(X_scenario)[:, 1]
    return pd_proba.mean()

def run_scenarios():
    print("Загружаю данные и модель...")
    # Загружаем модель
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_best_model.cbm"))
    
    # В run_scenarios():
    df_test = pd.read_csv(PROCESSED_DIR / "test_fe.csv")
    X_static = df_test.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')

    # Обязательно приводим категориальные колонки к строкам ЗАРАНЕЕ, до подстановки макро
    cat_cols = X_static.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X_static[col] = X_static[col].fillna('Missing').astype(str)
    
    # Оставляем только 5000 клиентов для скорости (или всех, если комп мощный)
    if len(X_static) > 5000:
        X_static = X_static.sample(5000, random_state=42)
        
    # Загружаем прогноз (нашу заглушку)
    forecast_df = pd.read_csv(PROCESSED_DIR / "macro_forecast.csv")
    
    # Берем средние значения за 2024 год для базового сценария
    base_macro = pd.DataFrame([{
        'repo_rate': forecast_df['repo_rate_forecast'].mean(),
        'czk_eur': forecast_df['czk_eur_forecast'].mean(),
        'unemployment': forecast_df['unemployment_forecast'].mean(),
        'px': forecast_df['px_forecast'].mean(),
        'vix': forecast_df['vix_forecast'].mean()
    }])
    
    # Оптимистичный сценарий (ставка падает, рынок растет)
    opt_macro = pd.DataFrame([{
        'repo_rate': 2.0, 'czk_eur': 24.0, 'unemployment': 2.0, 'px': 1800, 'vix': 12.0
    }])
    
    # Кризисный сценарий (ставка растет, паника)
    crisis_macro = pd.DataFrame([{
        'repo_rate': 8.0, 'czk_eur': 28.0, 'unemployment': 7.0, 'px': 800, 'vix': 65.0
    }])
    
    print("Считаю сценарии...")
    pd_opt = get_portfolio_pd(model, X_static, opt_macro)
    pd_base = get_portfolio_pd(model, X_static, base_macro)
    pd_crisis = get_portfolio_pd(model, X_static, crisis_macro)
    
    results = pd.DataFrame({
        'Сценарий': ['Оптимистичный', 'Базовый', 'Кризисный'],
        'PD (средний риск, %)': [round(pd_opt*100, 2), round(pd_base*100, 2), round(pd_crisis*100, 2)]
    })
    
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ СЦЕНАРНОГО АНАЛИЗА")
    print("=" * 40)
    print(results.to_string(index=False))
    
    return results

if __name__ == "__main__":
    run_scenarios()