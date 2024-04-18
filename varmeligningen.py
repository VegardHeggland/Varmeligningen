import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

alpha = 0.001 #Diffusitivitet
L = 30 #Lengde
t = 100 #Total tid
dx = 1
dt = (dx ** 2) / (4 * alpha)


gamma = (alpha * dt) / (dx ** 2)


x = np.linspace(-L/2, L/2, int(L/dx))
y = np.linspace(-L/2, L/2, int(L/dx))
X, Y = np.meshgrid(x, y)
T0 = np.exp(-(X**2 + Y**2)/50)

T = np.zeros((t, T0.shape[0], T0.shape[1]))

T[0,:,:] = T0

T[:, 0, :] = 0
T[:, -1, :] = 0
T[:, :, 0] = 0
T[:, :, -1] = 0


def calculate(u):
    for k in range(0, t - 1):
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                u[k + 1, i, j] = gamma * (u[k][i + 1][j] + u[k][i - 1][j] + u[k][i][j + 1] + u[k][i][j - 1] - 4 * u[k][i][j]) + u[k][i][j]
    return u

T = calculate(T)

x = np.arange(0, L, dx)
y = np.arange(0, L, dx)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def animate(k):
    ax.clear()
    ax.plot_surface(X, Y, T[k], cmap=plt.cm.jet, rstride=1, cstride=1, edgecolor='none')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Temperature')
    ax.set_title(f"Temperature Distribution")
    ax.set_zlim(0,1)


anim = FuncAnimation(fig, animate, frames=t, interval=50)
plt.show()