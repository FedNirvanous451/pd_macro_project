import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

def calculate_gini(y_true, y_proba):
    """Считает коэффициент Джини (Gini = 2 * ROC-AUC - 1)"""
    auc = roc_auc_score(y_true, y_proba)
    return 2 * auc - 1

def calculate_ks(y_true, y_proba):
    """Считает статистику Колмогорова-Смирнова (KS)"""
    # Разделяем вероятности на два массива: для класса 0 и для класса 1
    proba_class_0 = y_proba[y_true == 0]
    proba_class_1 = y_proba[y_true == 1]
    
    # Считаем KS
    ks_stat, p_value = ks_2samp(proba_class_0, proba_class_1)
    return ks_stat

def evaluate_model(y_true, y_proba, model_name="Model"):
    """Возвращает словарь со всеми метриками"""
    auc = roc_auc_score(y_true, y_proba)
    gini = calculate_gini(y_true, y_proba)
    ks = calculate_ks(y_true, y_proba)
    
    return {
        "Model": model_name,
        "ROC-AUC": round(auc, 4),
        "Gini": round(gini, 4),
        "KS": round(ks, 4)
    }