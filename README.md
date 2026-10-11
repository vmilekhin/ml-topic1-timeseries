# Тема 1: Прогнозирование USD/RUB

## Данные
USDRUB, дневные, 2020-01-02 — 2024-12-31 (1304 строки).
Train: 1043 строки (до 2024-01-01). Test: 261 строка (2024 год).

## Что проверено

### 1. ARIMA на ценах (`03_arima.py`)
Grid search (p,1,q), p,q∈[0..3] по AIC. Прогноз статический на тест.
Результат: MAE/RMSE/MAPE значительно хуже naive.

### 2. ARIMA на log-доходностях (`03b_arima_returns.py`)
- ADF: p-value = 0.0000 → ряд стационарен, d=0
- Grid search (p,0,q), p,q∈[0..5] по AIC
- Best: ARIMA(1,0,2), AIC = -5456.99
- Коэффициенты: const=0.00034, ar.L1=-0.353, ma.L1=0.379, ma.L2=0.172
  (AR/MA значимы, p < 0.01)
- Прогноз walk-forward one-step на 261 точке теста
- Ljung-Box: lag=10 p=0.12 (ОК), lag=20 p=0.0008

## Результат

| Модель | MAE (returns) | RMSE (returns) |
|---|---|---|
| Baseline (predict 0) | 0.008356 | 0.013274 |
| ARIMA(1,0,2) walk-forward | 0.008849 | 0.013672 |
| Улучшение | -5.9% | -3.0% |

| Модель | MAE (price) | RMSE (price) | MAPE (price) |
|---|---|---|---|
| Naive (p̂_t = p_{t-1}) | 0.7870 | 1.2888 | 0.834% |
| ARIMA(1,0,2) → price | 0.8332 | 1.3253 | 0.884% |
| Улучшение | -5.9% | -2.8% | -5.9% |

## Вывод

ARIMA(1,0,2) не превосходит naive-бейзлайн: она ухудшает метрики
на 3-6%. AR/MA-коэффициенты при этом статистически значимы, но
типичная величина прогноза r̂_t ~ 0.002 оказывается в ~5 раз меньше
типичной |r_t| ~ 0.010, и на метриках это не отражается.

Для USD/RUB это ожидаемо: дневные log-доходности валютных пар близки
к белому шуму. Negative result подтверждает корректность методики:
- ADF → grid search по AIC → walk-forward → Ljung-Box
- Метрики на доходностях + пересчёт в цены
- Baseline «predict 0» и naive p̂_t = p_{t-1}

Что не поможет дальше: ARIMA с большими p,q; ARMA; экспоненциальное
сглаживание. Что может помочь: GARCH на волатильность, экзогенные
признаки (нефть, ставка ЦБ, DXY).

## Файлы
- `01_eda.py` — EDA, ACF/PACF
- `02_clean.py`, `02b_clean_v2.py` — очистка данных
- `03_arima.py` — ARIMA на ценах
- `03b_arima_returns.py` — ARIMA на доходностях (walk-forward)
- `data/` — данные и метрики
- `figures/` — графики

## Воспроизведение
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python 01_eda.py
python 02b_clean_v2.py
python 03_arima.py
python 03b_arima_returns.py
