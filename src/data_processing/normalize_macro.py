"""
Нормализация макроэкономических факторов Чехии.

Читает 5 файлов:
- cnb.csv                          (repo rate CNB, месячные)
- eur_czk_2010_2023.csv            (курс EUR/CZK, дневные)
- czech_unemployment_2010_2023.csv (безработица, месячные)
- px.csv                           (индекс PX, дневные)
- vix.csv                          (VIX, дневные)

Приводит к единому формату: date | repo_rate | czk_eur | unemployment | px | vix
Агрегирует дневные данные в месячные (среднее за месяц).
Сохраняет результат в data/processed/macro_monthly.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# __file__ — это путь к текущему скрипту.
# .parents[0] — папка, где лежит скрипт (data_processing)
# .parents[1] — папка src
# .parents[2] — КОРЕНЬ ПРОЕКТА (поднялись на 2 директории выше)

ROOT_DIR = Path(__file__).resolve().parents[2]

# Теперь можно строить пути от корня:
RAW_DIR = ROOT_DIR / "data"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"
END = "2023-12-31"


# ============================================================
# 1. Repo rate CNB (месячные, уже готовые)
# ============================================================
def load_cnb():
    """
    cnb.csv: observation_date | IRSTCI01CZM156N
    Переименовываем в repo_rate и приводим дату.
    """
    print("Обрабатываю cnb.csv...")
    df = pd.read_csv(RAW_DIR / "cnb.csv")
    df = df.rename(columns={
        "observation_date": "date",
        "IRSTCI01CZM156N": "repo_rate",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} строк, период {df['date'].min().date()} — {df['date'].max().date()}")
    return df


# ============================================================
# 2. Курс EUR/CZK (дневные → месячные)
# ============================================================
def load_eur_czk():
    """
    eur_czk_2010_2023.csv: Date | EUR_CZK
    Агрегируем в месячные (среднее за месяц).
    """
    print(" Обрабатываю eur_czk_2010_2023.csv...")
    df = pd.read_csv(RAW_DIR / "eur_czk_2010_2023.csv")
    df = df.rename(columns={"Date": "date", "EUR_CZK": "czk_eur"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df.sort_values("date").reset_index(drop=True)

    # Агрегация в месячные
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["czk_eur"].mean().reset_index()
    monthly = monthly.rename(columns={"month": "date"})
    print(f"  {len(monthly)} месяцев (из {len(df)} дней)")
    return monthly


# ============================================================
# 3. Безработица (месячные, уже готовые)
# ============================================================
def load_unemployment():
    """
    czech_unemployment_2010_2023.csv: Date | Unemployment_Rate
    """
    print("Обрабатываю czech_unemployment_2010_2023.csv...")
    df = pd.read_csv(RAW_DIR / "czech_unemployment_2010_2023.csv")
    df = df.rename(columns={
        "Date": "date",
        "Unemployment_Rate": "unemployment",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} строк, период {df['date'].min().date()} — {df['date'].max().date()}")
    return df


# ============================================================
# 4. Индекс PX (дневные → месячные, требует очистки)
# ============================================================
def load_px():
    """
    px.csv: Date | Price | Open | High | Low | Vol. | Change %
    - Даты в формате MM/DD/YYYY
    - Цены с запятыми: '1,414.02'
    - Оставляем только Date + Price
    - Агрегируем в месячные
    """
    print("Обрабатываю px.csv...")
    df = pd.read_csv(RAW_DIR / "px.csv")

    # Оставляем только Date и Price
    df = df[["Date", "Price"]].copy()
    df = df.rename(columns={"Date": "date", "Price": "px"})

    # Чистим цены: убираем запятые, превращаем в float
    df["px"] = df["px"].astype(str).str.replace(",", "", regex=False)
    df["px"] = pd.to_numeric(df["px"], errors="coerce")

    # Чистим даты: MM/DD/YYYY → YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    # Убираем битые строки
    df = df.dropna(subset=["date", "px"])

    # Фильтр по периоду
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df.sort_values("date").reset_index(drop=True)

    # Агрегация в месячные
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["px"].mean().reset_index()
    monthly = monthly.rename(columns={"month": "date"})
    print(f"   {len(monthly)} месяцев (из {len(df)} дней)")
    return monthly


# ============================================================
# 5. VIX (дневные → месячные)
# ============================================================
def load_vix():
    """
    vix.csv: Date | VIX_Index
    Агрегируем в месячные (среднее за месяц).
    """
    print("Обрабатываю vix.csv...")
    df = pd.read_csv(RAW_DIR / "vix.csv")
    df = df.rename(columns={"Date": "date", "VIX_Index": "vix"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna(subset=["vix"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df.sort_values("date").reset_index(drop=True)

    # Агрегация в месячные
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["vix"].mean().reset_index()
    monthly = monthly.rename(columns={"month": "date"})
    print(f" {len(monthly)} месяцев (из {len(df)} дней)")
    return monthly


# ============================================================
# 6. Объединение всех факторов
# ============================================================
def merge_all():
    """
    Объединяет все 5 макро-факторов в один DataFrame по дате.
    """
    print("\n" + "=" * 60)
    print("ОБЪЕДИНЕНИЕ ВСЕХ МАКРО-ФАКТОРОВ")
    print("=" * 60)

    cnb = load_cnb()
    eur = load_eur_czk()
    unemp = load_unemployment()
    px = load_px()
    vix = load_vix()

    # Приводим все даты к первому дню месяца
    for df in [cnb, eur, unemp, px, vix]:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    # Поочерёдный merge
    merged = cnb.merge(eur, on="date", how="outer")
    merged = merged.merge(unemp, on="date", how="outer")
    merged = merged.merge(px, on="date", how="outer")
    merged = merged.merge(vix, on="date", how="outer")

    # Сортируем и заполняем пропуски
    merged = merged.sort_values("date").reset_index(drop=True)

    # Проверяем пропуски
    print("\nПропуски в объединённом датасете:")
    print(merged.isnull().sum())

    # Заполняем пропуски линейной интерполяцией (макро-ряды плавные)
    merged = merged.interpolate(method="linear", limit_direction="both")

    print(f"\nИтоговый датасет: {merged.shape}")
    print(f"   Период: {merged['date'].min().date()} — {merged['date'].max().date()}")
    print(f"   Колонки: {merged.columns.tolist()}")
    print("\nПервые 5 строк:")
    print(merged.head().to_string())
    print("\nПоследние 5 строк:")
    print(merged.tail().to_string())

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("НОРМАЛИЗАЦИЯ МАКРО-ФАКТОРОВ ЧЕХИИ")
    print("=" * 60 + "\n")

    macro = merge_all()

    # Сохраняем
    output_path = PROCESSED_DIR / "macro_monthly.csv"
    macro.to_csv(output_path, index=False)
    print(f"\nСохранено в: {output_path}")
    print("\nЭТАП 1 ЗАВЕРШЁН")