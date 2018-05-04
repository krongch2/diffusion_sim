from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
plt.rc('font', style='normal', family='serif',
    serif='Computer Modern Roman')
plt.rc('text', usetex=True)

def ring():
    array = np.zeros([nx])
    for i in range(nx):
        if 0.05 <= (i * dx - 0.5) ** 2 <= 0.1:
            array[i] = 1
    return array

def gaussian():
    array = np.zeros([nx])
    for i in range(nx):
        sigma_x = 0.1
        zx = (i * dx - 0.5) / sigma_x
        array[i] = 3 * np.exp(-zx ** 2 / 2)
    array[0] = 0
    array[-1] = 0
    return array

def source():
    array = np.zeros([nx])
    array[0] = 1
    array[-1] = 1
    return array

def bar():
    array = np.zeros([nx])
    for i in range(nx):
        if 0.2 <= i * dx <= 0.8:
            array[i] = 1
    return array

def well():
    array = np.ones([nx]) * 1000
    for i in range(nx):
        if 0.3 <= i * dx <= 0.7:
            array[i] = 0
    return array

def harmonic():
    array = np.zeros([nx])
    for i in range(nx):
        array[i] = 2 * (i * dx - 0.5) ** 2 / 0.5 ** 2
    return array

def line():
    array = np.zeros([nx])
    for i in range(nx):
        array[i] = 3 * (i * dx) ** 3
    return array

def zero():
    return np.zeros([nx])

def dirac():
    array = np.zeros([nx])
    array[int(nx / 2)] = 10000
    return array

def barrier():
    array = np.zeros([nx])
    for i in range(nx):
        if 0.4 <= i * dx <= 0.6:
            array[i] = 3
    return array

def pulse():
    array = np.zeros([nx])
    for i in range(nx):
        zx = (i * dx - 0.2) / 0.001
        array[i] = 0.5 * np.exp(-zx ** 2 / 2)
    array[0] = 0
    array[-1] = 0
    return array


##################################################################

def next(num):
    for repeat in range(20):
        uxx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx2
        u[1:-1] = u[1:-1] + dt * (k * uxx - V[1:-1] * u[1:-1])
        u[1:-1] = u[1:-1] / np.sqrt((u[1:-1] ** 2 * dx).sum())
    line.set_ydata(u)
    E = -(u[1:-1] * (k * uxx - V[1:-1] * u[1:-1]) * dx).sum()
    energy_text.set_text('Energy = {:.2f}'.format(E))
    return line, energy_text

def init(u_init, potential):
    filename = '{}_{}'.format(u_init.__name__, potential.__name__)
    return u_init(), potential(), filename

dx = 0.01
dx2 = dx ** 2
nx = int(1 / dx)
k = 0.5
dt = dx2 / (3 * k)
u, V, filename = init(pulse, barrier)

fig, ax = plt.subplots()
line, = ax.plot(np.arange(nx), V, color='#ef2c32', alpha=0.85, lw=1,
    marker='o', ms=3, mew=1, mec='#ef2c32', label='Potential')
line, = ax.plot(np.arange(nx), u, color='#4c72af', alpha=0.85, lw=1,
    marker='o', ms=3, mew=1, mec='#4c72af', label='Concentration')
ax.legend(bbox_to_anchor=(0.96, 0.93), fontsize=14)
ax.set_xlabel('x')
ax.set_ylim(0, 2.5)
energy_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ani = FuncAnimation(fig, next, frames=90, blit=False)
ani.save(filename + '.mp4', fps=15, bitrate=1600, dpi=600)
