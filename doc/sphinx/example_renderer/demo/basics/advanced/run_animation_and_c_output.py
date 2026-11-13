# %% [markdown]
# # Animations and C-level output
#
# Matplotlib animations left in the namespace are saved as mp4 (or embedded as
# an HTML widget when `ffmpeg` is missing). Output written directly to file
# descriptor 1, as Shamrock's C++ logger does, is captured as well.

# %%
import ctypes

libc = ctypes.CDLL(None)
libc.printf(b"hello from C printf\n")
print("hello from Python print")

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200)
(line,) = ax.plot(x, np.sin(x))


def update(frame):
    line.set_ydata(np.sin(x + 0.15 * frame))
    return (line,)


anim = FuncAnimation(fig, update, frames=40, interval=50, blit=True)
