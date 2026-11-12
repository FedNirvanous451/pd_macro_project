import pandas as pd
import numpy as np
import optuna
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Отключаем логи Optuna, чтобы не засорять консоль
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    print("Загружаю данные для тюнинга...")
    train = pd.read_csv(PROCESSED_DIR / "train_fe.csv")
    val = pd.read_csv(PROCESSED_DIR / "val_fe.csv")
    
    X_train = train.drop(columns=['TARGET', 'SK_ID_CURR'])
    y_train = train['TARGET']
    X_val = val.drop(columns=['TARGET', 'SK_ID_CURR'])
    y_val = val['TARGET']
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    return X_train, y_train, X_val, y_val, cat_cols, num_cols

# ==========================================
# 1. ПОДБОР ДЛЯ RANDOM FOREST
# ==========================================
def tune_random_forest(X_train, y_train, X_val, y_val, cat_cols, num_cols, n_trials=15):
    print(f"\nПодбор параметров для Random Forest ({n_trials} итераций)...")
    
    # Предобработка (OneHot для категорий)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), 
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train_prep, y_train)
        y_proba = model.predict_proba(X_val_prep)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Лучший ROC-AUC для RF: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")
    
    # Обучаем финальную модель с лучшими параметрами
    best_model = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1)
    best_model.fit(X_train_prep, y_train)
    
    # Сохраняем и модель, и препроцессор (чтобы потом применять)
    joblib.dump(best_model, MODELS_DIR / "rf_best_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "rf_preprocessor.pkl")
    print("Random Forest сохранен.")
    
    return study.best_value

# ==========================================
# 2. ПОДБОР ДЛЯ CATBOOST
# ==========================================
def tune_catboost(X_train, y_train, X_val, y_val, cat_cols, n_trials=15):
    print(f"\nПодбор параметров для CatBoost ({n_trials} итераций)...")
    
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].fillna('Missing').astype(str)
        X_val_cb[col] = X_val_cb[col].fillna('Missing').astype(str)
        
    cat_features_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols]

    def objective(trial):
        params = {
            'iterations': 500,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_seed': 42,
            'eval_metric': 'AUC',
            'verbose': False,
            'class_weights': [1, 3],
            'early_stopping_rounds': 50
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train_cb, y_train, cat_features=cat_features_idx, eval_set=(X_val_cb, y_val))
        y_proba = model.predict_proba(X_val_cb)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Лучший ROC-AUC для CatBoost: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")
    
    # Обучаем финальную модель
    best_params = study.best_params
    best_params['iterations'] = 1000 # Чуть больше итераций для финальной модели
    best_model = CatBoostClassifier(**best_params, class_weights=[1, 3], random_seed=42, verbose=False)
    best_model.fit(X_train_cb, y_train, cat_features=cat_features_idx)
    
    best_model.save_model(str(MODELS_DIR / "catboost_best_model.cbm"))
    print("CatBoost сохранен.")
    
    return study.best_value

if __name__ == "__main__":
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    print("=" * 60)
    print("ЭТАП 4.1: ПОДБОР ГИПЕРПАРАМЕТРОВ (OPTUNA)")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, cat_cols, num_cols = load_data()
    
    # Запускаем подбор (для экономии времени ставим n_trials=10-15)
    # В реальной жизни ставят 50-100
    rf_score = tune_random_forest(X_train, y_train, X_val, y_val, cat_cols, num_cols, n_trials=10)
    cb_score = tune_catboost(X_train, y_train, X_val, y_val, cat_cols, n_trials=10)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЮНИНГА")
    print("=" * 60)
    print(f"Random Forest (best ROC-AUC): {rf_score:.4f}")
    print(f"CatBoost (best ROC-AUC):      {cb_score:.4f}")
    print("\nЭТАП 4.1 ЗАВЕРШЕН")