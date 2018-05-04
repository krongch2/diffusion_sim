from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
plt.rc('font', style='normal', family='serif', serif='Computer Modern Roman')
plt.rc('text', usetex=True)

def ring():
    for i in range(nx):
        for j in range(ny):
            if 0.05 <= (i * dx - 0.5) ** 2 + (j * dy - 0.5) ** 2 <= 0.1:
                u[i, j] = 1

def gaussian():
    for i in range(nx):
        for j in range(ny):
            sigma_x = 0.1
            sigma_y = 0.1
            zx = (i * dx - 0.5) / sigma_x
            zy = (j * dy - 0.5) / sigma_y
            u[i, j] = np.exp(-zx ** 2 / 2 - zy ** 2 / 2)

def bound():
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

def edges():
    u[0:2, (nx / 2 - 2):(nx / 2 + 2)] = 1
    u[(nx / 2 -2):(nx / 2 + 2), 0:2] = 1

def const():
    u[:, :] = 0.5
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

def evolve(num):
    for repeat in range(10):
        uxx = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2
        uyy = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (k * (uxx + uyy) - V[1:-1, 1:-1] * u[1:-1, 1:-1])
        u[1:-1, 1:-1] = u[1:-1, 1:-1] / np.sqrt((u[1:-1, 1:-1] ** 2 * dx * dy).sum())
    print(num)
    image.set_array(u)
    return image

def harmonic():
    for i in range(nx):
        V[i, :] = 1-i * dx ** 2

def initial(func):
    func()
    return func.__name__ + '.mp4'

dx, dy = 0.01, 0.01
nx = int(1 / dx)
ny = int(1 / dy)
k = 0.5
dx2 = dx ** 2
dy2 = dy ** 2
dt = dx2 * dy2 / (2  * k * (dx2 + dy2))
u = np.zeros([nx, ny])
V = np.zeros([nx, ny])
filename = initial(ring)
harmonic()

fig = plt.figure()
image = plt.imshow(u, interpolation='nearest', cmap='hot', animated=True)
ani = FuncAnimation(fig, evolve, frames=90, blit=True)
plt.colorbar(image, ticks=[0, 1])
ani.save(filename, fps=15, bitrate=1600, dpi=600)
