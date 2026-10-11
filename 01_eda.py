"""Этап 1-2: Загрузка данных и разведочный анализ (EDA)."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

df = yf.download("RUB=X", start="2020-01-01", end="2025-01-01", progress=False)
df = df[["Close"]].copy()
df.columns = ["price"]
df = df.dropna()

print(f"Всего наблюдений: {len(df)}")
print(f"Период: {df.index[0].date()} — {df.index[-1].date()}")
print(f"\nПервые строки:\n{df.head()}")
print(f"\nПоследние строки:\n{df.tail()}")
print(f"\nОписательная статистика цены:\n{df['price'].describe()}")


# ---------------------------------------------------------------------------
# 2. Лог-доходности
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. ЛОГ-ДОХОДНОСТИ")
print("=" * 60)

# r_t = ln(P_t / P_{t-1})
df["log_return"] = np.log(df["price"] / df["price"].shift(1))
df = df.dropna()

print(f"Наблюдений после расчёта доходностей: {len(df)}")
print(f"\nСтатистика лог-доходностей:")
print(df["log_return"].describe())

print(f"\nСтандартное отклонение (волатильность): {df['log_return'].std():.6f}")
print(f"Годовая волатильность: {df['log_return'].std() * np.sqrt(252):.4f} "
      f"({df['log_return'].std() * np.sqrt(252) * 100:.2f}%)")


# ---------------------------------------------------------------------------
# 3. Графики
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. ГРАФИКИ")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Цена
axes[0].plot(df.index, df["price"], color="steelblue", linewidth=1)
axes[0].set_title("Курс USD/RUB", fontsize=14)
axes[0].set_ylabel("Рублей за 1 USD")
axes[0].grid(True, alpha=0.3)

# Лог-доходности
axes[1].plot(df.index, df["log_return"], color="darkred", linewidth=0.7)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Лог-доходности", fontsize=14)
axes[1].set_ylabel("log(P_t / P_{t-1})")
axes[1].grid(True, alpha=0.3)

# Гистограмма доходностей
axes[2].hist(df["log_return"], bins=80, color="darkred", alpha=0.7, edgecolor="black")
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("Распределение лог-доходностей", fontsize=14)
axes[2].set_xlabel("Лог-доходность")
axes[2].set_ylabel("Частота")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/01_price_returns.png", dpi=120, bbox_inches="tight")
print("Сохранено: figures/01_price_returns.png")
plt.close()


# ---------------------------------------------------------------------------
# 4. Тест на стационарность (ADF)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. ТЕСТ НА СТАЦИОНАРНОСТЬ (ADF)")
print("=" * 60)
print("H0: ряд нестационарен (есть единичный корень)")
print("H1: ряд стационарен")
print("Если p-value < 0.05 — отвергаем H0, ряд стационарен\n")

# Для цены
result_price = adfuller(df["price"], autolag="AIC")
print(f"ЦЕНА:")
print(f"  ADF-статистика: {result_price[0]:.4f}")
print(f"  p-value:        {result_price[1]:.6f}")
print(f"  Вывод:          {'стационарен' if result_price[1] < 0.05 else 'НЕстационарен'}")

# Для доходностей
result_ret = adfuller(df["log_return"], autolag="AIC")
print(f"\nЛОГ-ДОХОДНОСТИ:")
print(f"  ADF-статистика: {result_ret[0]:.4f}")
print(f"  p-value:        {result_ret[1]:.6f}")
print(f"  Вывод:          {'стационарен' if result_ret[1] < 0.05 else 'НЕстационарен'}")


# ---------------------------------------------------------------------------
# 5. ACF и PACF
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. ACF и PACF")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(df["log_return"], lags=40, ax=axes[0])
axes[0].set_title("ACF (автокорреляция) лог-доходностей", fontsize=13)

plot_pacf(df["log_return"], lags=40, ax=axes[1], method="ywm")
axes[1].set_title("PACF (частная автокорреляция) лог-доходностей", fontsize=13)

plt.tight_layout()
plt.savefig("figures/02_acf_pacf.png", dpi=120, bbox_inches="tight")
print("Сохранено: figures/02_acf_pacf.png")
plt.close()

print("\n" + "=" * 60)
print("ГОТОВО. Проверь графики в папке figures/")
print("=" * 60)
