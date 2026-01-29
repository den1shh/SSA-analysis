import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eth_usdt_3min.csv')
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
daily_close = df.groupby('date')['close'].last().reset_index()
prices = daily_close['close'].values.astype(np.float64)
y_train = prices[:400]
y_test = prices[400:]
steps = len(y_test)

N = len(y_train)
L = N // 2
K = N - L + 1

H = np.zeros((L, K), dtype=np.float64)
print(len(y_train), L, K)
for i in range(K):
    H[:, i] = y_train[i:i+L]

U, S, Vt = np.linalg.svd(H, full_matrices=False)
d = len(S)
X_elem = [np.sqrt(S[i]) * np.outer(U[:, i], Vt[i, :]) for i in range(d)]
ssn_idx = 35

trend = [i for i in range(1)]
season = [i for i in range(1, ssn_idx, 2)]
season_pairs = [[i, i+1] for i in season]
noise = [i for i in range(ssn_idx, d)]
groups = [trend] + season_pairs + [noise]


flat_indices = []
for g in groups:
    flat_indices.extend(g)
flat_indices = [i for i in flat_indices if 0 <= i < d]

def diagonal_averaging_to_N(Hmat, N_expected):
    Lh, Kh = Hmat.shape
    y = np.zeros(N_expected, dtype=Hmat.dtype)
    counts = np.zeros(N_expected, dtype=np.int64)
    for i in range(Lh):
        for j in range(Kh):
            idx = i + j
            if idx < N_expected:
                y[idx] += Hmat[i, j]
                counts[idx] += 1
    nonzero = counts != 0
    y[nonzero] /= counts[nonzero]
    return y

def compute_lrr_from_Ugroup(U_group, tol=1e-12):
    U_bar = U_group[:-1, :]
    pi = U_group[-1, :]
    nu2 = np.sum(pi**2)
    denom = 1.0 - nu2
    if denom <= tol:
        return None
    A = (U_bar @ pi) / denom
    return A

def forecast_by_A(y_init, A, steps):
    L_minus_1 = len(A)
    y = list(y_init)
    for _ in range(steps):
        block = y[-L_minus_1:]
        next_val = float(np.dot(A, block[::-1]))
        y.append(next_val)
    return np.array(y)

X_trend = sum(X_elem[i] for i in trend)
y_trend_recon = diagonal_averaging_to_N(X_trend, N_expected=N)

season_indices = [idx for pair in season_pairs for idx in pair if idx < d]
X_season = sum(X_elem[i] for i in season_indices) if season_indices else np.zeros_like(H)
y_season_recon = diagonal_averaging_to_N(X_season, N_expected=N)

X_noise = sum(X_elem[i] for i in noise)
y_noise_recon = diagonal_averaging_to_N(X_noise, N_expected=N)

# ---------------------------
# Прогнозирование компонент
# ---------------------------

# LRR для тренда
U_trend = np.column_stack([U[:, i] for i in trend])
A_trend = compute_lrr_from_Ugroup(U_trend)

# LRR для сезонности
U_season = np.column_stack([U[:, i] for i in season_indices]) if season_indices else np.zeros((L, 0))
A_season = compute_lrr_from_Ugroup(U_season) if season_indices else None

# Прогноз для тренда
if A_trend is not None:
    y_trend_full = forecast_by_A(y_trend_recon.tolist(), A_trend, steps)
    y_trend_forecast = y_trend_full[N:]  # Берем только прогноз
else:
    y_trend_forecast = np.zeros(steps)

delta_trend = y_trend_recon[-1] - y_trend_forecast[0]
y_trend_forecast += delta_trend

# Прогноз для сезонности
if A_season is not None:
    y_season_full = forecast_by_A(y_season_recon.tolist(), A_season, steps)
    y_season_forecast = y_season_full[N:]  # Берем только прогноз
else:
    y_season_forecast = np.zeros(steps)

delta_season = y_season_recon[-1] - y_season_forecast[0]
y_season_forecast += delta_season

forecast = y_trend_forecast + y_season_forecast 
forecast += (y_test[0] - forecast[0])

# ---------------------------
# Разложение тестовых данных на компоненты — стабильная версия (проекция на базис train)
# ---------------------------

# Надёжное формирование y_test_ext: последние last_len точек из train + все тестовые
last_len = N - len(y_test)
if last_len < 0:
    raise ValueError(f"len(y_test) ({len(y_test)}) больше N ({N}) — не могу собрать H_test с теми же размерами.")
if last_len == 0:
    y_test_ext = y_test.copy()
else:
    y_test_ext = np.concatenate([y_train[-last_len:], y_test])
# длина y_test_ext должна быть равна N (L+K-1)
assert len(y_test_ext) == N, "Ожидаемая длина y_test_ext = N"

# Построим Hankel для тестового блока (как и раньше)
H_test = np.zeros((L, K), dtype=np.float64)
for i in range(K):
    H_test[:, i] = y_test_ext[i:i+L]

# SVD теста (мы её оставляем только для сравнения; но для совместимой реконструкции будем делать проекцию)
U_test, S_test, Vt_test = np.linalg.svd(H_test, full_matrices=False)

# ---- Проекция H_test на базис train (U, S) ----
eps = 1e-12
# Выберем компоненты, которые хотим восстановить в train-базисе.
# Обычно хотим те же индексы, которые используем для тренда/сезонности:
use_indices = flat_indices.copy()  # например [0,1,2,3,...]
if len(use_indices) == 0:
    # защитная мера — если ничего не выбрано, используем первые r
    r_use = min(10, len(S))
    use_indices = list(range(r_use))

# определим максимально требуемый r чтобы проецировать (включая промежутки)
r_max = max(use_indices) + 1
if r_max > len(S):
    raise ValueError("Requested component index > available singular values")

# вычислим Vt_proj для первых r_max компонент: Vt = (1/S) * U.T @ H_test
K_test = H_test.shape[1]
Vt_proj = np.zeros((r_max, K_test), dtype=np.float64)
for i in range(r_max):
    if S[i] > eps:
        Vt_proj[i, :] = (U[:, i].T @ H_test) / S[i]
    else:
        Vt_proj[i, :] = 0.0

# Собираем проекцию отдельно для тренда и для сезонности (в train-базисе)
def assemble_proj(indices):
    Xp = np.zeros((L, K_test), dtype=np.float64)
    for i in indices:
        if i < r_max and S[i] > eps:
            Xp += np.sqrt(S[i]) * np.outer(U[:, i], Vt_proj[i, :])
    return Xp

# Тренд в train-базисе, реконструированная по H_test
X_trend_proj = assemble_proj(trend)
y_trend_proj_full = diagonal_averaging_to_N(X_trend_proj, N_expected=len(y_test_ext))
# тестовая часть — последние len(y_test) точек этой проекции
y_trend_test = y_trend_proj_full[-len(y_test):]

# Сезонность в train-базисе
X_season_proj = assemble_proj(season_indices)
y_season_proj_full = diagonal_averaging_to_N(X_season_proj, N_expected=len(y_test_ext))
y_season_test = y_season_proj_full[-len(y_test):]

# Для контроля можно также получить "чистую" SVD-на-тесте реконструкцию (как раньше),
# но для сравнения с forecast пользуемся проекцией на train-базис.
X_trend_test_svd = sum(np.sqrt(S_test[i]) * np.outer(U_test[:, i], Vt_test[i, :]) for i in trend)
y_trend_recon_test_svd = diagonal_averaging_to_N(X_trend_test_svd, N_expected=len(y_test_ext))[-len(y_test):]

# Теперь y_trend_test, y_season_test совместимы с тем, что предсказывает A_trend/A_season (train-база).

residuals = y_train - (y_trend_recon + y_season_recon)
noise_std = np.std(residuals)
scale = 0.5
np.random.seed(42)
gaussian_noise = np.random.normal(0, noise_std, len(forecast))*scale

# Прогноз с гауссовским шумом
forecast_with_noise = forecast + gaussian_noise

# ---------------------------
# Создаем одну картинку с ТРЕМЯ графиками
# ---------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

# Временные оси
t_train = np.arange(1, N+1)
t_test = np.arange(N+1, N + steps + 1)

# График 1: Тренд
ax1.plot(t_train, y_trend_recon, 'b-', linewidth=2, label='Train trend')
ax1.plot(t_test, y_trend_forecast, 'r--', linewidth=2, label='Test trend (forecast)')
ax1.axvline(x=N, color='k', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_ylabel('Trend')
ax1.set_title('Trend Component: Train vs Test vs Forecast')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Сезонность
ax2.plot(t_train, y_season_recon, 'b-', linewidth=2, label='Train seasonality')
ax2.plot(t_test, y_season_forecast, 'r--', linewidth=2, label='Test seasonality (forecast)')
ax2.axvline(x=N, color='k', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_ylabel('Seasonality')
ax2.set_title('Seasonality Component: Train vs Test vs Forecast')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Сумма trend + season
ax3.plot(t_train, y_train, 'b-', linewidth=2, label='Train (trend+season)')
ax3.plot(t_test, y_test, 'b-', linewidth=2, label='Train (trend+season)')
ax3.plot(t_test, forecast_with_noise, 'r--', linewidth=2, label='Test (trend+season forecast)')
ax3.axvline(x=N, color='k', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_ylabel('Trend + Season')
ax3.set_title('Combined Signal: Trend + Seasonality - Train vs Test vs Forecast')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.xlabel('Time Index')
plt.tight_layout()
plt.show()