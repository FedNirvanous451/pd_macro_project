import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

def get_portfolio_pd(model, X_static, macro_data):
    """Считает средний PD для портфеля клиентов при заданных макро-факторах."""
    X_scenario = X_static.copy()
    
    # Подставляем макро-факторы сценария
    for col in macro_data.columns:
        X_scenario[col] = float(macro_data[col].values[0])
        
    # Заполняем пропуски в текстовых колонках (как при обучении)
    cat_cols = X_scenario.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X_scenario[col] = X_scenario[col].fillna('Missing')
        
    return model.predict_proba(X_scenario)[:, 1].mean()

def run_multi_year_scenarios():
    print("=" * 60)
    print("ЭТАП 6: СЦЕНАРНЫЙ АНАЛИЗ И СТРЕСС-ТЕСТИРОВАНИЕ")
    print("=" * 60)
    
    print("Загружаю финальную модель и данные...")
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_final_model.cbm"))
    
    # Берем реальных клиентов из Test
    df_test = pd.read_csv(PROCESSED_DIR / "test_fe.csv")
    X_static = df_test.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    
    # Берем выборку 10 000 клиентов для скорости и репрезентативности
    if len(X_static) > 10000:
        X_static = X_static.sample(10000, random_state=42)
        
    print(f"Размер тестового портфеля: {len(X_static)} заемщиков")
    
    # Загружаем прогноз Prophet
    forecast_df = pd.read_csv(PROCESSED_DIR / "macro_forecast.csv")
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    
    years = [2024, 2025, 2026]
    base_scenarios = {}
    for year in years:
        year_data = forecast_df[forecast_df['date'].dt.year == year]
        base_scenarios[year] = pd.DataFrame([{
            'repo_rate': year_data['repo_rate_forecast'].mean(),
            'czk_eur': year_data['czk_eur_forecast'].mean(),
            'unemployment': year_data['unemployment_forecast'].mean(),
            'px': year_data['px_forecast'].mean(),
            'vix': year_data['vix_forecast'].mean()
        }])
        
    # Шоковые сценарии
    opt_macro = pd.DataFrame([{
        'repo_rate': 2.0, 'czk_eur': 24.0, 'unemployment': 2.0, 'px': 1800, 'vix': 12.0
    }])
    crisis_macro = pd.DataFrame([{
        'repo_rate': 8.0, 'czk_eur': 28.0, 'unemployment': 7.0, 'px': 800, 'vix': 65.0
    }])
    
    print("\nСчитаю сценарии...")
    
    pd_opt = get_portfolio_pd(model, X_static, opt_macro) * 100
    pd_crisis = get_portfolio_pd(model, X_static, crisis_macro) * 100
    
    results = []
    for year in years:
        pd_base = get_portfolio_pd(model, X_static, base_scenarios[year]) * 100
        results.append({
            'Год': year,
            'Оптимистичный (%)': round(pd_opt, 2),
            'Базовый прогноз (%)': round(pd_base, 2),
            'Кризисный (%)': round(pd_crisis, 2)
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("МАТРИЦА РИСКОВ (PD) ПО ГОДАМ")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Бизнес-интерпретация (Ожидаемые потери)
    print("\n" + "=" * 60)
    print("ОЖИДАЕМЫЕ ПОТЕРИ (Expected Loss)")
    print("=" * 60)
    
    # Предположим, что средний кредит = 500 000, а LGD = 45%
    portfolio_size = len(X_static) * 500_000
    lgd = 0.45
    
    pd_base_avg = results_df['Базовый прогноз (%)'].mean() / 100
    pd_crisis_val = pd_crisis / 100
    
    el_base = pd_base_avg * lgd * portfolio_size
    el_crisis = pd_crisis_val * lgd * portfolio_size
    
    print(f"Размер портфеля: {portfolio_size:,.0f}")
    print(f"Ожидаемые потери (Базовый сценарий): {el_base:,.0f}")
    print(f"Ожидаемые потери (Кризисный сценарий): {el_crisis:,.0f}")
    print(f"Дополнительные резервы в случае кризиса: {el_crisis - el_base:,.0f}")
    
    results_df.to_csv(PROCESSED_DIR / "scenario_results.csv", index=False)
    print("\nРезультаты сохранены в data/processed/scenario_results.csv")

if __name__ == "__main__":
    run_multi_year_scenarios()