import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os

# Данные находятся в папке data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eth_usdt_3min.csv')
df = pd.read_csv(data_path)
prices = df['close'].values.astype(np.float32)
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['date'] = df['timestamp'].dt.date

daily_close = df.groupby('date')['close'].last().reset_index()
prices = daily_close['close'].values.astype(np.float32)
y_train = prices[:400]

N = len(y_train)
L = N//2
K = N - L + 1 

H = np.zeros((L, K))
for i in range(K):
    H[:, i] = y_train[i:i+L]

U, S, Vt = np.linalg.svd(H, full_matrices=False)

num_vectors = 100  

odd_indices = list(range(1, num_vectors-1, 2))

fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], 'o-', markersize=4)
line2, = ax.plot([], [], 'o-', markersize=4)

ax.set_xlim(0, U.shape[0]-1)
ax.set_ylim(np.min(U[:, :num_vectors])-0.1, np.max(U[:, :num_vectors])+0.1)
ax.set_xlabel('Временной индекс в окне L')
ax.set_ylabel('Амплитуда')
ax.set_title('Попарная анимация левых сингулярных векторов SSA')
ax.grid(True)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update(frame):
    i = odd_indices[frame]
    line1.set_data(np.arange(U.shape[0]), U[:, i])
    line2.set_data(np.arange(U.shape[0]), U[:, i+1])
    
    line1.set_label(f'U_{i+1}')
    line2.set_label(f'U_{i+2}')
    ax.legend(loc='upper right')
    
    return line1, line2

ani = FuncAnimation(fig, update, frames=len(odd_indices), init_func=init,
                    blit=False, interval=750, repeat=True)

# Сохраняем GIF в папку figures
output_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'ssa_singular_vectors.gif')
ani.save(output_path, writer='pillow', fps=2)

plt.show()
