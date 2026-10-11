"""Шаг 3: ARIMA — подбор параметров, обучение, прогноз."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# 1. Загрузка почищенных данных
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv("data/usdrub_clean_v2.csv", index_col=0,
                 parse_dates=True)
df = df[["price", "log_return"]].dropna()

print(f"Строк: {len(df)}")
print(f"Период: {df.index[0].date()} — {df.index[-1].date()}")
print(f"Цена: {df['price'].min():.2f} — {df['price'].max():.2f}")


# ---------------------------------------------------------------------------
# 2. Train/Test split (по времени!)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. TRAIN/TEST SPLIT")
print("=" * 60)

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Train: {len(train)} ({train.index[0].date()} — {train.index[-1].date()})")
print(f"Test:  {len(test)} ({test.index[0].date()} — {test.index[-1].date()})")


# ---------------------------------------------------------------------------
# 3. Baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. BASELINE «завтра = сегодня»")
print("=" * 60)

baseline_pred = test["price"].shift(1)
baseline_true = test["price"]

# Метрики считаем только там, где есть baseline
mask = baseline_pred.notna()
baseline_mae = mean_absolute_error(
    baseline_true[mask], baseline_pred[mask])
baseline_rmse = np.sqrt(mean_squared_error(
    baseline_true[mask], baseline_pred[mask]))
baseline_mape = np.mean(np.abs(
    (baseline_true[mask] - baseline_pred[mask]) / baseline_true[mask])) * 100

print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"MAPE: {baseline_mape:.2f}%")


# ---------------------------------------------------------------------------
# 4. Grid search по (p, d, q)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. GRID SEARCH ПО (p, d, q)")
print("=" * 60)

# d=1, потому что цена нестационарна
d = 1
results = []

print(f"Пробую p ∈ [0..3], q ∈ [0..3] (d={d})...")

for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(train["price"], order=(p, d, q))
            fitted = model.fit()
            results.append({
                "p": p,
                "d": d,
                "q": q,
                "aic": fitted.aic,
                "bic": fitted.bic,
            })
        except Exception as e:
            continue

results_df = pd.DataFrame(results).sort_values("aic")
print(f"\nТоп-5 моделей по AIC:")
print(results_df.head(5).to_string(index=False))

best = results_df.iloc[0]
best_order = (int(best["p"]), int(best["d"]), int(best["q"]))
print(f"\nЛучшая модель: ARIMA{best_order}, AIC={best['aic']:.2f}")


# ---------------------------------------------------------------------------
# 5. Обучение лучшей модели на train
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. ОБУЧЕНИЕ НА TRAIN")
print("=" * 60)

model = ARIMA(train["price"], order=best_order)
fitted = model.fit()

print(fitted.summary().tables[0])
print("\nПараметры:")
print(fitted.params)


# ---------------------------------------------------------------------------
# 6. Прогноз на test
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. ПРОГНОЗ НА TEST")
print("=" * 60)

forecast = fitted.forecast(steps=len(test))
forecast = pd.Series(forecast.values, index=test.index)

arima_mae = mean_absolute_error(test["price"], forecast)
arima_rmse = np.sqrt(mean_squared_error(test["price"], forecast))
arima_mape = np.mean(np.abs((test["price"] - forecast) / test["price"])) * 100

print(f"MAE:  {arima_mae:.4f}")
print(f"RMSE: {arima_rmse:.4f}")
print(f"MAPE: {arima_mape:.2f}%")


# ---------------------------------------------------------------------------
# 7. Сравнение с baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("7. СРАВНЕНИЕ С BASELINE")
print("=" * 60)

print(f"{'Метрика':10s} {'Baseline':>12s} {'ARIMA':>12s} {'Улучшение':>12s}")
print("-" * 50)
for name, b, a in [("MAE", baseline_mae, arima_mae),
                   ("RMSE", baseline_rmse, arima_rmse),
                   ("MAPE", baseline_mape, arima_mape)]:
    improvement = (b - a) / b * 100
    sign = "✓" if improvement > 0 else "✗"
    print(f"{name:10s} {b:>12.4f} {a:>12.4f} {improvement:>10.1f}% {sign}")


# ---------------------------------------------------------------------------
# 8. Графики
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("8. ГРАФИКИ")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Прогноз на тесте
axes[0].plot(train.index, train["price"], color="steelblue",
             linewidth=1, label="Train")
axes[0].plot(test.index, test["price"], color="black",
             linewidth=1, label="Test (реальность)")
axes[0].plot(test.index, forecast, color="darkred",
             linewidth=1, label=f"ARIMA{best_order}")
baseline_pred = test["price"].shift(1)
baseline_true = test["price"]

# Метрики считаем только там, где есть baseline
mask = baseline_pred.notna()
baseline_mae = mean_absolute_error(
    baseline_true[mask], baseline_pred[mask])
baseline_rmse = np.sqrt(mean_squared_error(
    baseline_true[mask], baseline_pred[mask]))
baseline_mape = np.mean(np.abs(
    (baseline_true[mask] - baseline_pred[mask]) / baseline_true[mask])) * 100

axes[0].set_title("Прогноз ARIMA на тесте")
axes[0].set_ylabel("Рублей за 1 USD")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Остатки
residuals = fitted.resid
axes[1].plot(residuals.index, residuals, color="darkred", linewidth=0.7)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Остатки модели (должны быть похожи на белый шум)")
axes[1].set_ylabel("Остаток")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/05_arima.png", dpi=120, bbox_inches="tight")
plt.close()
print("Сохранено: figures/05_arima.png")


# ---------------------------------------------------------------------------
# 9. Сохранение результатов
# ---------------------------------------------------------------------------

results_df.to_csv("data/arima_grid_results.csv", index=False)
print("Сохранено: data/arima_grid_results.csv")

# Сохраняем метрики для сравнения с ML
metrics = {
    "model": "ARIMA",
    "order": str(best_order),
    "mae": arima_mae,
    "rmse": arima_rmse,
    "mape": arima_mape,
}
pd.DataFrame([metrics]).to_csv("data/arima_metrics.csv", index=False)
print("Сохранено: data/arima_metrics.csv")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)
