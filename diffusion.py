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
    return 'ring.mp4'

def gaussian():
    for i in range(nx):
        for j in range(ny):
            sigma_x = 0.1
            sigma_y = 0.1
            zx = (i * dx - 0.5) / sigma_x
            zy = (j * dy - 0.5) / sigma_y
            u[i, j] = np.exp(-zx ** 2 / 2 - zy ** 2 / 2)
    return 'gaussian.mp4'

def random():
    for i in range(5, nx, 5):
        for j in range(5, ny, 5):
            u[i, j] = np.random.random_sample()
    return 'random.mp4'

def spots():
    for i in range(5, nx, 5):
        for j in range(5, ny, 5):
            u[i, j] = 1
    return 'spots.mp4'

def bound():
    u[0, :] = 1
    u[:, 0] = 1
    return 'bound.mp4'

def edges():
    u[0:2, (nx / 2 - 2):(nx / 2 + 2)] = 1
    u[(nx / 2 -2):(nx / 2 + 2), 0:2] = 1
    return 'edges.mp4'

def evolve(*args):
    for k in range(8):
        uxx = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2
        uyy = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (uxx + uyy)
    image.set_array(u)
    print(la.norm(u))
    return image

dx, dy = 0.01, 0.01
nx = int(1 / dx)
ny = int(1 / dy)
dx2 = dx ** 2
dy2 = dy ** 2
dt = dx2 * dy2 / (2 * (dx2 + dy2))
u = np.zeros([nx, ny])
filename = ring()

fig = plt.figure()
image = plt.imshow(u, interpolation='nearest', cmap='hot', animated=True)
ani = FuncAnimation(fig, evolve, frames=90, blit=True)
plt.colorbar(image, ticks=[0, 1])
ani.save(filename, fps=15, bitrate=1600, dpi=600)
