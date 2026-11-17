import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

def train_final_model():
    print("=" * 60)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ CATBOOST (С ЛУЧШИМИ ПАРАМЕТРАМИ)")
    print("=" * 60)

    # 1. Загружаем данные
    print("Загружаю данные с фичами...")
    train = pd.read_csv(PROCESSED_DIR / "train_fe.csv")
    val = pd.read_csv(PROCESSED_DIR / "val_fe.csv")
    
    X_train = train.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    y_train = train['TARGET']
    X_val = val.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    y_val = val['TARGET']

    # 2. Находим категориальные колонки и заполняем в них пропуски
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing')
        X_val[col] = X_val[col].fillna('Missing')

    # 3. Загружаем лучшие параметры от Optuna
    params_path = MODELS_DIR / "best_catboost_params.csv"
    if params_path.exists():
        print("Загружаю лучшие параметры от Optuna...")
        best_params_df = pd.read_csv(params_path)
        best_params = best_params_df.iloc[0].to_dict()
        print(f"Параметры: {best_params}")
    else:
        print("ВНИМАНИЕ: Файл с параметрами не найден. Использую дефолтные.")
        best_params = {'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3.0}

    # 4. Инициализируем и обучаем финальную модель
    print("\nОбучаю финальную модель...")
    model = CatBoostClassifier(
        iterations=1000, # Даем больше итераций, так как есть early stopping
        learning_rate=best_params.get('learning_rate', 0.1),
        depth=int(best_params.get('depth', 6)),
        l2_leaf_reg=best_params.get('l2_leaf_reg', 3.0),
        eval_metric='AUC',
        random_seed=42,
        verbose=100, # Показывать прогресс каждые 100 итераций
        class_weights=[1, 3], # Компенсация дисбаланса классов
        early_stopping_rounds=50,
        cat_features=cat_cols # Указываем текстовые колонки
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # 5. Сохраняем
    model_path = MODELS_DIR / "catboost_final_model.cbm"
    model.save_model(str(model_path))
    print(f"\nМодель успешно обучена и сохранена в: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_final_model()