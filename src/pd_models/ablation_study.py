import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def calculate_metrics(y_true, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    gini = 2 * auc - 1
    # KS
    proba_0 = y_proba[y_true == 0]
    proba_1 = y_proba[y_true == 1]
    ks = ks_2samp(proba_0, proba_1).statistic
    return auc, gini, ks

def train_and_evaluate(X_train, y_train, X_val, y_val, cat_cols, name):
    print(f"\nОбучение модели: {name}...")
    
    # Заполняем пропуски в категориях
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing')
        X_val[col] = X_val[col].fillna('Missing')
        
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.1, depth=5, 
        eval_metric='AUC', random_seed=42, verbose=0, 
        class_weights=[1, 3], cat_features=cat_cols
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    auc, gini, ks = calculate_metrics(y_val, y_proba)
    
    print(f"  Результат: ROC-AUC = {auc:.4f}, Gini = {gini:.4f}, KS = {ks:.4f}")
    return {"Модель": name, "ROC-AUC": round(auc, 4), "Gini": round(gini, 4), "KS": round(ks, 4)}

if __name__ == "__main__":
    print("=" * 60)
    print("ABLATION STUDY: ВЛИЯНИЕ FEATURE ENGINEERING")
    print("=" * 60)
    
    # Загружаем данные С новыми фичами
    train = pd.read_csv(PROCESSED_DIR / "train_fe.csv")
    val = pd.read_csv(PROCESSED_DIR / "val_fe.csv")
    
    # Список наших новых фичей
    new_features = ['DTI', 'PTI', 'CREDIT_TERM_YEARS', 'AGE_YEARS', 'EMPLOYED_YEARS', 'INCOME_PER_PERSON', 'EXT_SOURCE_MEAN']
    lag_features = ['vix_lag_1', 'repo_lag_1']
    
    results = []
    
    # 1. BASELINE (убираем все новые фичи)
    train_base = train.drop(columns=new_features + lag_features, errors='ignore')
    val_base = val.drop(columns=new_features + lag_features, errors='ignore')
    
    X_train_base = train_base.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    y_train = train_base['TARGET']
    X_val_base = val_base.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    y_val = val_base['TARGET']
    
    cat_cols = X_train_base.select_dtypes(include=['object']).columns.tolist()
    
    results.append(train_and_evaluate(X_train_base.copy(), y_train, X_val_base.copy(), y_val, cat_cols, "1. Baseline (без новых фичей)"))
    
    # 2. С БАЗОВЫМИ ФИЧАМИ (DTI, PTI и т.д.)
    train_fe = train.drop(columns=lag_features, errors='ignore')
    val_fe = val.drop(columns=lag_features, errors='ignore')
    
    X_train_fe = train_fe.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    X_val_fe = val_fe.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    
    results.append(train_and_evaluate(X_train_fe.copy(), y_train, X_val_fe.copy(), y_val, cat_cols, "2. + Базовый FE (DTI, PTI и др.)"))
    
    # 3. С ЛАГАМИ МАКРО-ФАКТОРОВ
    X_train_all = train.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    X_val_all = val.drop(columns=['TARGET', 'SK_ID_CURR', 'issue_date'], errors='ignore')
    
    results.append(train_and_evaluate(X_train_all.copy(), y_train, X_val_all.copy(), y_val, cat_cols, "3. + Лаги макро (VIX, ставка)"))
    
    # Итоговая таблица
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА (ABLATION STUDY)")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Сохраняем в CSV для отчета
    results_df.to_csv(PROCESSED_DIR / "ablation_study_results.csv", index=False)
    print("\nРезультаты сохранены в data/processed/ablation_study_results.csv")