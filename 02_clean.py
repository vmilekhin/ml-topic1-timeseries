"""Шаг 2: Поиск и чистка выбросов в данных USD/RUB."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# 1. Загрузка
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

df = yf.download("RUB=X", start="2020-01-01", end="2025-01-01",
                 progress=False)
df = df[["Close"]].copy()
df.columns = ["price"]
df = df.dropna()

print(f"Всего строк: {len(df)}")
print(f"Период: {df.index[0].date()} — {df.index[-1].date()}")
print(f"min цена: {df['price'].min():.2f}")
print(f"max цена: {df['price'].max():.2f}")


# ---------------------------------------------------------------------------
# 2. Поиск выбросов
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. ПОИСК ВЫБРОСОВ")
print("=" * 60)

# Аномальные цены (реальный курс USD/RUB в 2020-2024: 55-130)
LOW_BOUND = 30.0
HIGH_BOUND = 200.0

mask = (df["price"] < LOW_BOUND) | (df["price"] > HIGH_BOUND)
outliers = df[mask]

print(f"\nАномальных цен (< {LOW_BOUND} или > {HIGH_BOUND}): {len(outliers)}")
if len(outliers) > 0:
    print(outliers)

# Экстремальные доходности
df["log_return"] = np.log(df["price"] / df["price"].shift(1))
extreme = df[df["log_return"].abs() > 0.3]
print(f"\nЭкстремальных доходностей (|r| > 0.3): {len(extreme)}")
if len(extreme) > 0:
    print(extreme[["price", "log_return"]])


# ---------------------------------------------------------------------------
# 3. Чистка
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. ЧИСТКА ВЫБРОСОВ")
print("=" * 60)

# Заменяем аномальные цены на NaN
df_clean = df.copy()
df_clean.loc[mask, "price"] = np.nan

n_replaced = mask.sum()
print(f"Заменено на NaN: {n_replaced}")

# Интерполяция
df_clean["price"] = df_clean["price"].interpolate(method="linear")
df_clean = df_clean.dropna()

print(f"Строк после чистки: {len(df_clean)}")
print(f"min цена: {df_clean['price'].min():.2f}")
print(f"max цена: {df_clean['price'].max():.2f}")


# ---------------------------------------------------------------------------
# 4. Пересчёт доходностей
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. ДОХОДНОСТИ ПОСЛЕ ЧИСТКИ")
print("=" * 60)

df_clean["log_return"] = np.log(
    df_clean["price"] / df_clean["price"].shift(1)
)
df_clean = df_clean.dropna()

print(f"Всего доходностей: {len(df_clean)}")
print(f"Среднее:  {df_clean['log_return'].mean():.6f}")
print(f"Std:      {df_clean['log_return'].std():.6f}")
print(f"Min:      {df_clean['log_return'].min():.6f}")
print(f"Max:      {df_clean['log_return'].max():.6f}")
print(f"\nГодовая волатильность: "
      f"{df_clean['log_return'].std() * np.sqrt(252):.4f} "
      f"({df_clean['log_return'].std() * np.sqrt(252) * 100:.2f}%)")


# ---------------------------------------------------------------------------
# 5. ADF-тест
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. ADF-ТЕСТ (после чистки)")
print("=" * 60)

result_price = adfuller(df_clean["price"], autolag="AIC")
result_ret = adfuller(df_clean["log_return"], autolag="AIC")

print(f"Цена:")
print(f"  ADF-статистика: {result_price[0]:.4f}")
print(f"  p-value:        {result_price[1]:.6f}")
print(f"  Вывод:          "
      f"{'стационарна' if result_price[1] < 0.05 else 'НЕстационарна'}")

print(f"\nДоходности:")
print(f"  ADF-статистика: {result_ret[0]:.4f}")
print(f"  p-value:        {result_ret[1]:.6f}")
print(f"  Вывод:          "
      f"{'стационарны' if result_ret[1] < 0.05 else 'НЕстационарны'}")


# ---------------------------------------------------------------------------
# 6. Графики
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. ГРАФИКИ")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df.index, df["price"], color="lightcoral",
             linewidth=1, alpha=0.6, label="До чистки")
axes[0].plot(df_clean.index, df_clean["price"], color="steelblue",
             linewidth=1, label="После чистки")
axes[0].set_title("Курс USD/RUB: до и после чистки")
axes[0].set_ylabel("Рублей за 1 USD")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df_clean.index, df_clean["log_return"],
             color="darkred", linewidth=0.7)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Лог-доходности (после чистки)")
axes[1].set_ylabel("log(P_t / P_{t-1})")
axes[1].grid(alpha=0.3)

axes[2].hist(df_clean["log_return"], bins=80,
             color="darkred", alpha=0.7, edgecolor="black")
axes[2].set_title("Распределение доходностей (после чистки)")
axes[2].set_xlabel("Лог-доходность")
axes[2].set_ylabel("Частота")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/03_clean_data.png", dpi=120, bbox_inches="tight")
plt.close()
print("Сохранено: figures/03_clean_data.png")


# ---------------------------------------------------------------------------
# 7. Сохранение чистых данных
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("7. СОХРАНЕНИЕ")
print("=" * 60)

df_clean.to_csv("data/usdrub_clean.csv")
print("Сохранено: data/usdrub_clean.csv")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)
