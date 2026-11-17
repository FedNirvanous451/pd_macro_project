import warnings
import logging
from pathlib import Path
import pandas as pd
from prophet import Prophet

# Отключаем спам в консоли
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def forecast_factor(df, col_name, best_params, periods):
    print(f"  Прогнозирую {col_name}...")
    
    # Подготовка данных для Prophet
    p_df = df[['date', col_name]].copy()
    p_df = p_df.rename(columns={'date': 'ds', col_name: 'y'})
    
    # Сглаживаем ступенчатые ряды (как при обучении)
    if col_name in ['repo_rate', 'unemployment']:
        p_df['y'] = p_df['y'].rolling(window=3, min_periods=1).mean()
    
    # Достаем лучшие параметры для этого фактора
    params = best_params[col_name]
    
    # Инициализируем модель с лучшими параметрами
    m = Prophet(
        yearly_seasonality=(col_name in ['vix', 'px', 'czk_eur']),
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale']
    )
    
    m.fit(p_df)
    
    # Делаем прогноз на 36 месяцев вперед (2024 год-2026год)
    future = m.make_future_dataframe(periods=periods, freq='MS')
    forecast = m.predict(future)
    
    # Берем только прогнозные месяцы и нужные колонки
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    
    # Переименовываем под наш формат
    result = result.rename(columns={
        'ds': 'date', 
        'yhat': f'{col_name}_forecast',
        'yhat_lower': f'{col_name}_lower',
        'yhat_upper': f'{col_name}_upper'
    })
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("ЭТАП 5.2: ФИНАЛЬНЫЙ ПРОГНОЗ МАКРО-ФАКТОРОВ (PROPHET)")
    print("=" * 60)
    
    # Загружаем исторические данные
    df = pd.read_csv(PROCESSED_DIR / "macro_monthly.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Загружаем параметры, найденные Optuna
    best_params_df = pd.read_csv(PROCESSED_DIR / "prophet_best_params.csv", index_col=0)
    best_params = best_params_df.to_dict(orient='index')
    
    factors = ['repo_rate', 'czk_eur', 'unemployment', 'px', 'vix']
    forecast_dfs = []
    
    for factor in factors:
        f_df = forecast_factor(df, factor, best_params, periods=36)
        forecast_dfs.append(f_df)
        
    # Объединяем прогнозы в один датафрейм
    final_forecast = forecast_dfs[0]
    for f_df in forecast_dfs[1:]:
        final_forecast = final_forecast.merge(f_df, on='date', how='left')
        
    # Округляем для красоты
    numeric_cols = final_forecast.select_dtypes(include=['float64']).columns
    final_forecast[numeric_cols] = final_forecast[numeric_cols].round(2)
        
    # Сохраняем
    out_path = PROCESSED_DIR / "macro_forecast.csv"
    final_forecast.to_csv(out_path, index=False)
    
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ ПРОГНОЗ НА 2024-2026 ГОДЫ:")
    print("=" * 60)
    
    # Выводим только точечные прогнозы
    display_cols = ['date'] + [f'{f}_forecast' for f in factors]
    print(final_forecast[display_cols].head(3).to_string(index=False))
    print("...")
    print(final_forecast[display_cols].tail(3).to_string(index=False))
    
    print(f"\nВсе результаты (с доверительными интервалами) сохранены в: {out_path}")
    print("ЭТАП 5 ЗАВЕРШЕН")