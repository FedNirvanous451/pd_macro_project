from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def create_features(df):
    df = df.copy()
    
    # 1. Долговая нагрузка
    df['DTI'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['PTI'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # 2. Срок кредита (в годах)
    df['CREDIT_TERM_YEARS'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'] / 12
    
    # 3. Возраст и стаж в годах (переводим из отрицательных дней)
    df['AGE_YEARS'] = df['DAYS_BIRTH'].abs() / 365.25
    # В DAYS_EMPLOYED есть аномалия 365243 (пенсионеры), обрабатываем ее
    df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x == 365243 else abs(x)) / 365.25
    
    # 4. Доход на члена семьи
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    
    # 5. Средний внешний скоринг (заполняем пропуски нулями, если нет оценок)
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    
    # 6. Лаги макро-факторов (сдвиг на 1 месяц назад для каждого клиента)
    # Сортируем по дате, чтобы лаги считались правильно
    if 'issue_date' in df.columns:
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        df = df.sort_values('issue_date')
        
        df['vix_lag_1'] = df['vix'].shift(1)
        df['repo_lag_1'] = df['repo_rate'].shift(1)
        df['vix_lag_1'] = df['vix_lag_1'].bfill() # Заполняем первый пропуск
        df['repo_lag_1'] = df['repo_lag_1'].bfill()

    return df

if __name__ == "__main__":
    print("Выполняю Feature Engineering...")
    
    for split in ['train', 'val', 'test']:
        file_path = PROCESSED_DIR / f"{split}_clean.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df_fe = create_features(df)
            
            # Сохраняем
            out_path = PROCESSED_DIR / f"{split}_fe.csv"
            df_fe.to_csv(out_path, index=False)
            print(f"  {split}: добавлено фичей. Итоговый размер: {df_fe.shape}")
            
    print("Feature Engineering завершен. Файлы сохранены.")