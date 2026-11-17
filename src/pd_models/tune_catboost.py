import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    print("Загружаю данные с новыми фичами...")
    train = pd.read_csv(PROCESSED_DIR / "train_fe.csv")
    val = pd.read_csv(PROCESSED_DIR / "val_fe.csv")
    
    X_train = train.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    y_train = train['TARGET']
    X_val = val.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    y_val = val['TARGET']
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    # Заполняем пропуски в категориях
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing')
        X_val[col] = X_val[col].fillna('Missing')
        
    return X_train, y_train, X_val, y_val, cat_cols

def tune_catboost(X_train, y_train, X_val, y_val, cat_cols, n_trials=15):
    print(f"\nПодбор параметров для CatBoost ({n_trials} итераций)...")
    
    def objective(trial):
        params = {
            'iterations': 500,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_seed': 42,
            'eval_metric': 'AUC',
            'verbose': False,
            'class_weights': [1, 3],
            'early_stopping_rounds': 50,
            'cat_features': cat_cols
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Лучший ROC-AUC для CatBoost: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")
    
    # Сохраняем лучшие параметры в файл, чтобы потом использовать в final_model.py
    pd.DataFrame([study.best_params]).to_csv(MODELS_DIR / "best_catboost_params.csv", index=False)
    print("Параметры сохранены в models/best_catboost_params.csv")
    
    return study.best_value

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, cat_cols = load_data()
    best_score = tune_catboost(X_train, y_train, X_val, y_val, cat_cols, n_trials=15)