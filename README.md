<div align="center">

# 🏦 PD-Macro Analytics

### Прогнозирование вероятности дефолта заёмщиков с учётом макроэкономических факторов Чехии

**Прогнозно-аналитическая система для оценки кредитного риска и стресс-тестирования**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-FFCC00?style=for-the-badge&logo=yandex&logoColor=black)](https://catboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prefect](https://img.shields.io/badge/Prefect-2.14+-070E10?style=for-the-badge&logo=prefect&logoColor=white)](https://www.prefect.io/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)](./LICENSE)

[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-passing-success?style=flat-square)]()
[![Code style](https://img.shields.io/badge/Code%20style-black-000000?style=flat-square)]()

</div>

---

## 📖 О проекте

**PD-Macro Analytics** — это end-to-end система машинного обучения, которая прогнозирует **вероятность дефолта (PD)** заёмщиков с учётом **макроэкономической динамики Чешской Республики**.

Система отвечает на практический вопрос банковского риск-менеджмента:

> *«Как изменится вероятность дефолта заёмщиков, если ставка ЦБ вырастет, курс кроны ослабнет, а мировой рынок войдёт в кризис?»*

### 🎯 Ключевые возможности

| | Возможность | Описание |
|---|---|---|
| 📥 | **Сбор данных** | 5 разнородных источников (Kaggle, CNB, PX, Yahoo, Eurostat) |
| 🧹 | **Обработка** | Очистка, работа с пропусками, генерация признаков (DTI, PTI, лаги) |
| 🔮 | **Прогноз макро** | Prophet / SARIMA / Naive → выбор лучшего по MAPE |
| 🤖 | **PD-модели** | LogReg / Random Forest / CatBoost → выбор лучшего по ROC-AUC |
| 🧠 | **Интерпретация** | SHAP-анализ для каждого заёмщика |
| 🎬 | **Сценарии** | Оптимистичный / базовый / кризисный стресс-тест |
| 📊 | **Дашборд** | Streamlit с ползунками макро-факторов в реальном времени |
| ⚙️ | **Оркестрация** | Prefect для автоматического запуска пайплайна |

---

## 🏗️ Архитектура системы

```mermaid
flowchart TD
    A[📥 Источники данных] --> B[⚙️ ETL-модуль]
    B --> C[(🗄️ Хранилище)]
    C --> D[🧹 Обработка и EDA]
    D --> E[🔮 Прогноз макро]
    D --> F[🤖 PD-модели]
    E --> G[📈 Прогноз PD]
    F --> G
    G --> H[🎬 Сценарии]
    H --> I[📊 Дашборд]
    
    A1[Home Credit] --> A
    A2[CNB] --> A
    A3[PX / Yahoo] --> A
    A4[VIX] --> A
    A5[Eurostat] --> A
    
    style A fill:#DAE8FC,stroke:#6C8EBF
    style B fill:#D5E8D4,stroke:#82B366
    style C fill:#FFF2CC,stroke:#D6B656
    style D fill:#D5E8D4,stroke:#82B366
    style E fill:#FFE6CC,stroke:#D79B00
    style F fill:#FFE6CC,stroke:#D79B00
    style G fill:#FFE6CC,stroke:#D79B00
    style H fill:#E1D5E7,stroke:#9673A6
    style I fill:#E1D5E7,stroke:#9673A6