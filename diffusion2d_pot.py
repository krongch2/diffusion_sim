from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
plt.rc('font', style='normal', family='serif',
    serif='Computer Modern Roman')
plt.rc('text', usetex=True)

def ring():
    array = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            if 0.05 <= (i * dx - 0.5) ** 2 + (j * dy - 0.5) ** 2 <= 0.1:
                array[i, j] = 1
    return array

def gaussian():
    array = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            sigma_x = 0.1
            sigma_y = 0.1
            zx = (i * dx - 0.5) / sigma_x
            zy = (j * dy - 0.5) / sigma_y
            array[i, j] = np.exp(-zx ** 2 / 2 - zy ** 2 / 2)
    return array

def source():
    array = np.zeros([nx, ny])
    array[0, :] = 1
    array[-1, :] = 1
    array[:, 0] = 1
    array[:, -1] = 1
    return array

def bar():
    array = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            if 0.2 <= i * dx <= 0.8:
                array[i, j] = 1
    return array

def well():
    array = np.ones([nx, ny]) * 1000
    for i in range(nx):
        for j in range(ny):
            if 0.3 <= i * dx <= 0.7 and 0.3 <= j * dy <= 0.7:
                array[i, j] = 0
    return array

def harmonic():
    array = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            array[i, j] = 2 * (i * dx - 0.5) ** 2 / 0.5 ** 2 + 2 * (j * dy - 0.5) ** 2 / 0.5 ** 2
    return array

# def line():
#     array = np.zeros([nx, ny])
#     for i in range(nx):
#         for j in range(ny):
#             array[i] = 3 * (i * dx) ** 3
#     return array

def zero():
    return np.zeros([nx, ny])

def dirac():
    array = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            array[int(nx / 2), int(ny / 2)] = 10000
    return array

##################################################################

def next(frame):
    for repeat in range(10):
        uxx = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2
        uyy = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (k * (uxx + uyy) - V[1:-1, 1:-1] * u[1:-1, 1:-1])
        u[1:-1, 1:-1] = u[1:-1, 1:-1] / np.sqrt((u[1:-1, 1:-1] ** 2 * dx * dy).sum())
    print(frame)
    image_u.set_array(u)
    E = -(u[1:-1, 1:-1] * (k * uxx - V[1:-1, 1:-1] * u[1:-1, 1:-1]) * dx).sum()
    energy_text.set_text('Energy = {:.2f}'.format(E))
    return image_u, energy_text

def init(u_init, potential):
    filename = '{}_{}'.format(u_init.__name__, potential.__name__)
    return u_init(), potential(), filename

dx, dy = 0.01, 0.01
dx2, dy2 = dx ** 2, dy ** 2
nx, ny = int(1 / dx), int(1 / dy)
k = 0.5
dt = dx2 * dy2 / (4  * k * (dx2 + dy2))
u, V, filename = init(ring, well)

fig, axes = plt.subplots(nrows=1, ncols=2)
image_u = axes[0].imshow(u, interpolation='nearest', cmap='hot', animated=True, origin='lower')
axes[0].set_title('Concentration')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(image_u, ax=axes[0], ticks=[0, 1])

image_V = axes[1].imshow(V, interpolation='nearest', cmap='gist_earth', animated=True, origin='lower')
axes[1].set_title('Potential')
axes[1].set_xlabel('x')
fig.colorbar(image_V, ax=axes[1], ticks=[0, 4])

energy_text = axes[0].text(0.22, -0.4, '', transform=axes[0].transAxes)

ani = FuncAnimation(fig, next, frames=180, blit=False)
ani.save(filename + '.mp4', fps=15, bitrate=1600, dpi=600)
