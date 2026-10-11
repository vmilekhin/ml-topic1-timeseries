"""Шаг 2b: Усиленная чистка — аномальные цены И экстремальные доходности."""

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
print("1. ЗАГРУЗКА")
print("=" * 60)

df = yf.download("RUB=X", start="2020-01-01", end="2025-01-01",
                 progress=False)
df = df[["Close"]].copy()
df.columns = ["price"]
df = df.dropna()

print(f"Строк: {len(df)}")


# ---------------------------------------------------------------------------
# 2. Итеративная чистка
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. ИТЕРАТИВНАЯ ЧИСТКА")
print("=" * 60)

# Границы для цены USD/RUB
LOW = 30.0
HIGH = 200.0
# Порог для доходности
RET_THRESHOLD = 0.15  # 15% в день — уже подозрительно

df_clean = df.copy()

for iteration in range(5):
    # 2.1. Аномальные цены
    price_mask = (df_clean["price"] < LOW) | (df_clean["price"] > HIGH)
    n_price = price_mask.sum()
    df_clean.loc[price_mask, "price"] = np.nan

    # 2.2. Экстремальные доходности
    ret = np.log(df_clean["price"] / df_clean["price"].shift(1))
    ret_mask = ret.abs() > RET_THRESHOLD
    n_ret = ret_mask.sum()

    # Заменяем экстремальные цены на NaN
    # (берём обе точки — и саму, и предыдущую)
    extreme_indices = df_clean.index[ret_mask.fillna(False)]

    for idx in extreme_indices:
        pos = df_clean.index.get_loc(idx)
        if pos > 0:
            df_clean.iloc[pos, df_clean.columns.get_loc("price")] = np.nan
            df_clean.iloc[pos - 1, df_clean.columns.get_loc("price")] = np.nan

    # Интерполяция
    df_clean["price"] = df_clean["price"].interpolate(method="linear")

    print(f"Итерация {iteration + 1}: "
          f"аномальных цен = {n_price}, "
          f"экстремальных доходностей = {n_ret}")

    if n_price == 0 and n_ret == 0:
        print("  Больше выбросов нет.")
        break

df_clean = df_clean.dropna()


# ---------------------------------------------------------------------------
# 3. Пересчёт доходностей
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. ДОХОДНОСТИ ПОСЛЕ ЧИСТКИ")
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
      f"{df_clean['log_return'].std() * np.sqrt(252) * 100:.2f}%")


# ---------------------------------------------------------------------------
# 4. ADF-тест
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. ADF-ТЕСТ")
print("=" * 60)

r_price = adfuller(df_clean["price"], autolag="AIC")
r_ret = adfuller(df_clean["log_return"], autolag="AIC")

print(f"Цена:       p-value = {r_price[1]:.6f} → "
      f"{'стационарна' if r_price[1] < 0.05 else 'НЕстационарна'}")
print(f"Доходности: p-value = {r_ret[1]:.6f} → "
      f"{'стационарны' if r_ret[1] < 0.05 else 'НЕстационарны'}")


# ---------------------------------------------------------------------------
# 5. Графики
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. ГРАФИКИ")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df.index, df["price"], color="lightcoral",
             linewidth=1, alpha=0.5, label="До чистки")
axes[0].plot(df_clean.index, df_clean["price"], color="steelblue",
             linewidth=1, label="После чистки")
axes[0].set_title("USD/RUB: до и после усиленной чистки")
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
axes[2].set_title("Распределение доходностей")
axes[2].set_xlabel("Лог-доходность")
axes[2].set_ylabel("Частота")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/04_clean_v2.png", dpi=120, bbox_inches="tight")
plt.close()
print("Сохранено: figures/04_clean_v2.png")


# ---------------------------------------------------------------------------
# 6. Сохранение
# ---------------------------------------------------------------------------

df_clean.to_csv("data/usdrub_clean_v2.csv")
print("\nСохранено: data/usdrub_clean_v2.csv")
print(f"Строк в файле: {len(df_clean)}")
