import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Данные находятся в папке data
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
for i in range(K):
    H[:, i] = y_train[i:i+L]

U, S, Vt = np.linalg.svd(H, full_matrices=False)
d = len(S)
X_elem = [np.sqrt(S[i]) * np.outer(U[:, i], Vt[i, :]) for i in range(d)]

trend = [0]
season = [i for i in range(1, min(75, d-1), 2)]
season_pairs = [[i, i+1] for i in season]
groups = [trend] + season_pairs

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

if len(flat_indices) == 0:
    X_selected = np.zeros_like(H)
else:
    X_selected = sum(X_elem[i] for i in flat_indices)

y_selected_recon = diagonal_averaging_to_N(X_selected, N_expected=N)

if len(flat_indices) > 0:
    U_group = np.column_stack([U[:, i] for i in flat_indices])
    A = compute_lrr_from_Ugroup(U_group)
else:
    U_group = np.zeros((L, 0))
    A = None

if A is not None:
    y_full = forecast_by_A(y_selected_recon.tolist(), A, steps)
    y_forecast = y_full[N:]  
else:
    y_full = None
    y_forecast = None

plt.figure(figsize=(12,6))

t_train = np.arange(0, N)
plt.plot(t_train, y_selected_recon, '-', color='tab:blue', linewidth=2, label='selected groups (recon)')

if y_forecast is not None:
    t_fore = np.arange(N, N + len(y_forecast))
    plt.plot(t_fore, y_forecast, '--', color='tab:blue', linewidth=2, label='selected groups (forecast)')

plt.axvline(x=N-1, color='k', linestyle='--', linewidth=1)
plt.xlim(0, N + steps)
plt.xlabel('Time index')
plt.ylabel('Value')
plt.title('Суммарный сигнал от trend + season (recon и forecast)')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.02))
plt.grid(True)
plt.tight_layout()
plt.show()
