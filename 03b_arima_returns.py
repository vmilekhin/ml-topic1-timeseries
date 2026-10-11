"""Шаг 3b: ARIMA на log-доходностях — корректный подход.

Отличие от 03_arima.py:
- модель строится на log_return (d=0), а не на цене (d=1)
- walk-forward one-step forecast вместо статического на 20% теста
- метрики на доходностях + пересчёт в цены
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Загрузка
print("=" * 60)
print("1. ЗАГРУЗКА")
print("=" * 60)

df = pd.read_csv("data/usdrub_clean_v2.csv", index_col=0, parse_dates=True)
df = df[["price", "log_return"]].dropna()

print(f"Строк: {len(df)}")
print(f"Период: {df.index[0].date()} — {df.index[-1].date()}")


# 2. ADF на доходностях
print("\n" + "=" * 60)
print("2. ADF ТЕСТ НА log_return")
print("=" * 60)

adf_stat, adf_p, *_ = adfuller(df["log_return"], autolag="AIC")
print(f"ADF statistic: {adf_stat:.4f}")
print(f"p-value:       {adf_p:.4f}")
if adf_p < 0.05:
    print("→ Ряд стационарен (p < 0.05), используем d=0")
else:
    print("→ Ряд НЕ стационарен, стоит подумать про d=1")


# 3. Train/test split
print("\n" + "=" * 60)
print("3. TRAIN/TEST SPLIT")
print("=" * 60)

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

print(f"Train: {len(train)} ({train.index[0].date()} — {train.index[-1].date()})")
print(f"Test:  {len(test)} ({test.index[0].date()} — {test.index[-1].date()})")


# 4. Baseline: random walk на доходностях = predict 0
print("\n" + "=" * 60)
print("4. BASELINE (predict 0 для доходностей)")
print("=" * 60)

b_mae = mean_absolute_error(test["log_return"], np.zeros(len(test)))
b_rmse = np.sqrt(mean_squared_error(test["log_return"], np.zeros(len(test))))
print(f"MAE:  {b_mae:.6f}")
print(f"RMSE: {b_rmse:.6f}")


# 5. Grid search по (p, 0, q)
print("\n" + "=" * 60)
print("5. GRID SEARCH (p, 0, q)")
print("=" * 60)

results = []
for p in range(6):
    for q in range(6):
        if p == 0 and q == 0:
            continue
        try:
            m = ARIMA(train["log_return"], order=(p, 0, q)).fit()
            results.append({"p": p, "q": q, "aic": m.aic, "bic": m.bic})
        except Exception:
            continue

results_df = pd.DataFrame(results).sort_values("aic")
print("Топ-5 по AIC:")
print(results_df.head(5).to_string(index=False))

best = results_df.iloc[0]
best_order = (int(best["p"]), 0, int(best["q"]))
print(f"\nЛучшая: ARIMA{best_order}, AIC={best['aic']:.2f}")


# 6. Walk-forward one-step forecast
print("\n" + "=" * 60)
print("6. WALK-FORWARD ONE-STEP FORECAST")
print("=" * 60)

history = list(train["log_return"].values)
predictions = []

for i, true_r in enumerate(test["log_return"].values):
    try:
        m = ARIMA(history, order=best_order).fit()
        yhat = float(m.forecast(steps=1).iloc[0])
    except Exception:
        yhat = 0.0
    predictions.append(yhat)
    history.append(true_r)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(test)}")

pred = pd.Series(predictions, index=test.index)


# 7. Метрики на доходностях
print("\n" + "=" * 60)
print("7. МЕТРИКИ НА ДОХОДНОСТЯХ")
print("=" * 60)

a_mae = mean_absolute_error(test["log_return"], pred)
a_rmse = np.sqrt(mean_squared_error(test["log_return"], pred))

print(f"{'Метрика':8s} {'Baseline(0)':>14s} {'ARIMA':>14s} {'Улучш.':>10s}")
print("-" * 52)
for name, b, a in [("MAE", b_mae, a_mae), ("RMSE", b_rmse, a_rmse)]:
    imp = (b - a) / b * 100
    print(f"{name:8s} {b:>14.6f} {a:>14.6f} {imp:>9.1f}%")


# 8. Пересчёт в цены
print("\n" + "=" * 60)
print("8. ПЕРЕСЧЁТ В ЦЕНЫ")
print("=" * 60)

last_train_price = train["price"].iloc[-1]
prices = []
prev = last_train_price
for r in pred.values:
    prev = prev * np.exp(r)
    prices.append(prev)
price_pred = pd.Series(prices, index=test.index)

naive_price = test["price"].shift(1)
naive_price.iloc[0] = last_train_price

price_mae = mean_absolute_error(test["price"], price_pred)
price_rmse = np.sqrt(mean_squared_error(test["price"], price_pred))
price_mape = np.mean(np.abs((test["price"] - price_pred) / test["price"])) * 100

n_mae = mean_absolute_error(test["price"], naive_price)
n_rmse = np.sqrt(mean_squared_error(test["price"], naive_price))
n_mape = np.mean(np.abs((test["price"] - naive_price) / test["price"])) * 100

print(f"{'Метрика':8s} {'Naive':>14s} {'ARIMA':>14s} {'Улучш.':>10s}")
print("-" * 52)
for name, n, a in [("MAE", n_mae, price_mae),
                   ("RMSE", n_rmse, price_rmse),
                   ("MAPE", n_mape, price_mape)]:
    imp = (n - a) / n * 100
    print(f"{name:8s} {n:>14.4f} {a:>14.4f} {imp:>9.1f}%")


# 9. Диагностика остатков
print("\n" + "=" * 60)
print("9. LJUNG-BOX НА ОСТАТКАХ")
print("=" * 60)

fitted_full = ARIMA(train["log_return"], order=best_order).fit()
lb = acorr_ljungbox(fitted_full.resid, lags=[10, 20], return_df=True)
print(lb)


# 10. Графики
print("\n" + "=" * 60)
print("10. ГРАФИКИ")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 10.1 Прогноз доходностей
axes[0, 0].plot(test.index, test["log_return"],
                color="black", linewidth=0.8, label="Реальные")
axes[0, 0].plot(test.index, pred,
                color="darkred", linewidth=0.8, label=f"ARIMA{best_order}")
axes[0, 0].axhline(0, color="gray", linewidth=0.5)
axes[0, 0].set_title("log_return: прогноз vs реальность")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 10.2 Прогноз цен
axes[0, 1].plot(train.index[-100:], train["price"].iloc[-100:],
                color="steelblue", linewidth=1, label="Train (хвост)")
axes[0, 1].plot(test.index, test["price"],
                color="black", linewidth=1, label="Test")
axes[0, 1].plot(test.index, price_pred,
                color="darkred", linewidth=1, label="ARIMA (пересчёт)")
axes[0, 1].plot(test.index, naive_price,
                color="gray", linewidth=0.8, linestyle="--", label="Naive")
axes[0, 1].set_title("Цена: прогноз vs naive")
axes[0, 1].set_ylabel("RUB за 1 USD")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 10.3 Остатки
axes[1, 0].plot(fitted_full.resid.index, fitted_full.resid,
                color="darkred", linewidth=0.6)
axes[1, 0].axhline(0, color="black", linewidth=0.5)
axes[1, 0].set_title("Остатки на train")
axes[1, 0].grid(alpha=0.3)

# 10.4 Гистограмма остатков
axes[1, 1].hist(fitted_full.resid, bins=40,
                color="steelblue", edgecolor="white")
axes[1, 1].set_title("Распределение остатков")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/06_arima_returns.png", dpi=120, bbox_inches="tight")
plt.close()
print("Сохранено: figures/06_arima_returns.png")


# 11. Сохранение
results_df.to_csv("data/arima_returns_grid.csv", index=False)
pd.DataFrame([{
    "model": "ARIMA_returns",
    "order": str(best_order),
    "mae_returns": a_mae,
    "rmse_returns": a_rmse,
    "mape_price": price_mape,
    "mae_price": price_mae,
    "rmse_price": price_rmse,
}]).to_csv("data/arima_returns_metrics.csv", index=False)
print("Сохранено: data/arima_returns_metrics.csv")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)
