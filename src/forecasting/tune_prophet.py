import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Отключаем лишний спам в консоли
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def objective(trial, df_train, df_val, col_name):
    # Optuna подбирает параметры для каждого фактора отдельно
    cps = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
    sps = trial.suggest_float('seasonality_prior_scale', 0.1, 20.0, log=True)
    
    # Подготовка данных для Prophet
    p_train = df_train[['date', col_name]].rename(columns={'date': 'ds', col_name: 'y'})
    p_val = df_val[['date', col_name]].rename(columns={'date': 'ds', col_name: 'y'})
    
    # Сглаживаем ступенчатые ряды (ставка ЦБ и безработица)
    if col_name in ['repo_rate', 'unemployment']:
        p_train['y'] = p_train['y'].rolling(window=3, min_periods=1).mean()
    
    # Инициализация модели (Монте-Карло включено по умолчанию)
    m = Prophet(
        yearly_seasonality=(col_name in ['vix', 'px', 'czk_eur']),
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps
    )
    
    try:
        m.fit(p_train)
        
        # Прогноз на период Validation
        future = m.make_future_dataframe(periods=len(p_val), freq='MS')
        forecast = m.predict(future)
        
        # Считаем ошибку только на Validation (2023 год)
        y_pred = forecast['yhat'].tail(len(p_val)).values
        y_true = p_val['y'].values
        
        # MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        return mape
    except Exception as e:
        # Если модель упадет (например, из-за математики), возвращаем огромную ошибку
        return 100.0

if __name__ == "__main__":
    print("=" * 60)
    print("ЭТАП 5.1: ПОДБОР ПАРАМЕТРОВ PROPHET (OPTUNA)")
    print("=" * 60)
    
    # Загружаем данные
    df = pd.read_csv(PROCESSED_DIR / "macro_monthly.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Делим на Train (до 2023) и Val (2023)
    df_train = df[df['date'].dt.year < 2023].copy()
    df_val = df[df['date'].dt.year == 2023].copy()
    
    factors = ['repo_rate', 'czk_eur', 'unemployment', 'px', 'vix']
    best_params = {}
    
    for factor in factors:
        print(f"\nПодбираю параметры для {factor}...")
        study = optuna.create_study(direction='minimize')
        
        # 15 итераций достаточно, чтобы найти хорошие параметры
        study.optimize(lambda trial: objective(trial, df_train, df_val, factor), n_trials=45)
        
        print(f"  Лучший MAPE: {study.best_value:.2%}")
        print(f"  Параметры: {study.best_params}")
        best_params[factor] = study.best_params
        
    # Сохраняем результаты
    params_df = pd.DataFrame(best_params).T
    out_path = PROCESSED_DIR / "prophet_best_params.csv"
    params_df.to_csv(out_path)
    
    print(f"\nПараметры сохранены в {out_path}")
    print("ЭТАП 5.1 ЗАВЕРШЕН")