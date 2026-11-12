"""
Подготовка кредитных данных и привязка макро-факторов.
Используются таблицы: application_train, previous_application, bureau.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Пути к директориям
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Константы
RANDOM_STATE = 42
ANCHOR_DATE = pd.Timestamp('2018-01-01') # Примерная дата выгрузки датасета

def load_and_prepare_main():
    print("Загружаю train.csv...")
    df = pd.read_csv(RAW_DIR / "train.csv")
    
    # Реконструируем дату на основе DAYS_ID_PUBLISH
    # Заполняем пропуски медианой, если они есть
    df['DAYS_ID_PUBLISH'] = df['DAYS_ID_PUBLISH'].fillna(df['DAYS_ID_PUBLISH'].median())
    
    # Переводим отрицательные дни в положительные и вычитаем из опорной даты
    days_ago = df['DAYS_ID_PUBLISH'].abs()
    df['issue_date'] = ANCHOR_DATE - pd.to_timedelta(days_ago, unit='D')
    
    # Округляем до начала месяца, чтобы совпадало с macro_monthly
    df['issue_date'] = df['issue_date'].dt.to_period('M').dt.to_timestamp()
    
    return df

def process_previous_application():
    print("Обрабатываю previous_application.csv...")
    prev = pd.read_csv(RAW_DIR / "previous_application.csv")
    
    # Создаем бинарные флаги для агрегации
    prev['IS_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev['IS_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    # Агрегация по клиенту
    agg_prev = prev.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'AMT_APPLICATION': ['mean', 'max'],
        'IS_APPROVED': ['sum', 'mean'],
        'IS_REFUSED': ['sum', 'mean'],
        'DAYS_DECISION': ['max', 'min']
    })
    
    # Упрощаем названия колонок
    agg_prev.columns = ['PREV_' + '_'.join(col).strip() for col in agg_prev.columns]
    agg_prev = agg_prev.reset_index()
    
    print(f"  Получено признаков previous_application: {agg_prev.shape[1] - 1}")
    return agg_prev

def process_bureau():
    print("Обрабатываю bureau.csv...")
    bureau = pd.read_csv(RAW_DIR / "bureau.csv")
    
    # Бинарные флаги
    bureau['IS_ACTIVE'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau['HAS_OVERDUE'] = (bureau['AMT_CREDIT_SUM_OVERDUE'] > 0).astype(int)
    
    # Агрегация по клиенту
    agg_bureau = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'IS_ACTIVE': 'sum',
        'AMT_CREDIT_SUM': 'sum',
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'max'],
        'HAS_OVERDUE': 'sum',
        'DAYS_CREDIT': 'max'
    })
    
    agg_bureau.columns = ['BUREAU_' + '_'.join(col).strip() for col in agg_bureau.columns]
    agg_bureau = agg_bureau.reset_index()
    
    print(f"  Получено признаков bureau: {agg_bureau.shape[1] - 1}")
    return agg_bureau

def merge_with_macro(credit_df):
    print("Привязываю макро-факторы...")
    macro_df = pd.read_csv(PROCESSED_DIR / "macro_monthly.csv")
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    
    merged = credit_df.merge(macro_df, left_on='issue_date', right_on='date', how='left')
    merged = merged.drop(columns=['date'])
    
    # Заполняем пропуски макро-факторов (если даты вышли за пределы 2010-2023)
    macro_cols = ['repo_rate', 'czk_eur', 'unemployment', 'px', 'vix']
    merged[macro_cols] = merged[macro_cols].ffill().bfill()
    
    return merged

def split_data(df):
    print("Разбиваю на Train / Val / Test СТРОГО ПО ВРЕМЕНИ (без перемешивания)...")
    
    # Сортируем по дате от старых к новым
    df = df.sort_values('issue_date').reset_index(drop=True)
    
    # Жесткие границы (с учетом того, как легли реконструированные даты)
    # Предполагаем, что данные до 2015 года - это старое, 2016-2017 - валидация, 2018 - тест
    train_df = df[df['issue_date'].dt.year <= 2015].copy()
    val_df = df[(df['issue_date'].dt.year >= 2016) & (df['issue_date'].dt.year <= 2017)].copy()
    test_df = df[df['issue_date'].dt.year == 2018].copy()
    
    # Если вдруг 2018 года оказалось слишком мало (даты уплыли), берем последние 15% строк
    if len(test_df) < 10000:
        print("  Мало данных за 2018 год, отрезаю последние 15% по времени...")
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()

    print(f"  Train: {train_df.shape}, период: {train_df['issue_date'].min().date()} — {train_df['issue_date'].max().date()}")
    print(f"  Val:   {val_df.shape}, период: {val_df['issue_date'].min().date()} — {val_df['issue_date'].max().date()}")
    print(f"  Test:  {test_df.shape}, период: {test_df['issue_date'].min().date()} — {test_df['issue_date'].max().date()}")
    
    print(f"  Доля дефолтов в Train: {train_df['TARGET'].mean():.4f}")
    print(f"  Доля дефолтов в Val:   {val_df['TARGET'].mean():.4f}")
    print(f"  Доля дефолтов в Test:  {test_df['TARGET'].mean():.4f}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    print("=" * 60)
    print("ЭТАП 2: ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)
    
    # 1. Загрузка основной таблицы
    main_df = load_and_prepare_main()
    
    # 2. Агрегация исторических данных
    prev_features = process_previous_application()
    bureau_features = process_bureau()
    
    # 3. Склейка таблиц
    print("Склеиваю таблицы...")
    full_df = main_df.merge(prev_features, on='SK_ID_CURR', how='left')
    full_df = full_df.merge(bureau_features, on='SK_ID_CURR', how='left')
    
    # 4. Обработка пропусков после merge
    # Заполняем нулями ТОЛЬКО новые признаки, чтобы не исказить оригинальные данные
    new_cols = [c for c in full_df.columns if c.startswith('PREV_') or c.startswith('BUREAU_')]
    full_df[new_cols] = full_df[new_cols].fillna(0)
    
    # 5. Привязка макро-факторов
    full_df = merge_with_macro(full_df)
    
    # 6. Разбиение на выборки
    train, val, test = split_data(full_df)
    
    # 7. Сохранение результатов
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val_processed.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)
    
    print("\nВсе файлы сохранены в data/processed/")
    print("ЭТАП 2 ЗАВЕРШЕН")