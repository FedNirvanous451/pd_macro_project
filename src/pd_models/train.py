import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Импортируем наши метрики
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.pd_models.metrics import evaluate_model

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

def load_data():
    print("Загружаю очищенные данные...")
    train = pd.read_csv(PROCESSED_DIR / "train_fe.csv")
    val = pd.read_csv(PROCESSED_DIR / "val_fe.csv")
    
    X_train = train.drop(columns=['TARGET', 'SK_ID_CURR'])
    y_train = train['TARGET']
    X_val = val.drop(columns=['TARGET', 'SK_ID_CURR'])
    y_val = val['TARGET']
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    return X_train, y_train, X_val, y_val, cat_cols, num_cols

if __name__ == "__main__":
    print("=" * 60)
    print("ЭТАП 4.2: ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, cat_cols, num_cols = load_data()
    results = []
    
    # 1. LOGISTIC REGRESSION (Обучаем здесь и сейчас)
    print("\nОбучение Logistic Regression...")
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    lr_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_val)[:, 1]
    results.append(evaluate_model(y_val, lr_proba, "Logistic Regression"))

    # 2. RANDOM FOREST (Загружаем лучшую из Optuna)
    print("\nЗагружаю лучшую модель Random Forest...")
    try:
        rf_model = joblib.load(MODELS_DIR / "rf_best_model.pkl")
        rf_preprocessor = joblib.load(MODELS_DIR / "rf_preprocessor.pkl")
        
        X_val_prep = rf_preprocessor.transform(X_val)
        rf_proba = rf_model.predict_proba(X_val_prep)[:, 1]
        results.append(evaluate_model(y_val, rf_proba, "Random Forest (Tuned)"))
    except FileNotFoundError:
        print("Ошибка: Файлы Random Forest не найдены. Запусти tune_models.py!")

    # 3. CATBOOST (Загружаем лучшую из Optuna)
    print("\nЗагружаю лучшую модель CatBoost...")
    try:
        cb_model = CatBoostClassifier()
        cb_model.load_model(str(MODELS_DIR / "catboost_best_model.cbm"))
        
        # Для CatBoost нужно заполнить пропуски в категориях, как мы делали при обучении
        X_val_cb = X_val.copy()
        for col in cat_cols:
            X_val_cb[col] = X_val_cb[col].fillna('Missing').astype(str)
            
        cb_proba = cb_model.predict_proba(X_val_cb)[:, 1]
        results.append(evaluate_model(y_val, cb_proba, "CatBoost (Tuned)"))
    except FileNotFoundError:
        print("Ошибка: Файл CatBoost не найден. Запусти tune_models.py!")

    # ВЫВОД РЕЗУЛЬТАТОВ
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ (на Validation выборке)")
    print("=" * 60)
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
    else:
        print("Не удалось собрать результаты.")
    
    print("\nЭТАП 4 ЗАВЕРШЕН")