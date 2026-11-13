# %% [markdown]
# # Plots, outputs and tags
#
# This page was generated from a plain Python script in the jupytext percent
# format. Prose is **Markdown** (MyST), so roles such as
# {math}`\rho_a = \sum_b m_b W_{ab}` work.
#
# ## Setup
#
# ### Parameters

# %%
# shamdoc_thumbnail_number = 2
import matplotlib.pyplot as plt
import numpy as np

dr = 0.1
bsize = 4
print(f"dr = {dr}, bsize = {bsize}")

# %% [markdown]
# The last expression of a cell is displayed, like in a notebook:

# %%
np.arange(5) * dr

# %% [markdown]
# ```{note}
# Directives work as usual in Markdown cells.
# ```
#
# $$
# \rho_a = \sum_b m_b W(r_{ab}, h_a)
# $$
#
# ## Plots
#
# ### A first figure

# %%
x = np.linspace(-bsize, bsize, 200)
plt.figure()
plt.plot(x, np.exp(-(x**2)))
plt.xlabel("x")
plt.ylabel(r"$\rho$")
plt.show()

# %% [markdown]
# ### A second figure (used as thumbnail)
#
# The plotting code is collapsed with the `hide-input` tag.

# %% tags=["hide-input"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
axs[0].plot(x, np.sin(x))
axs[1].plot(x, np.cos(x))
plt.show()

# %% [rst]
# Cells marked ``[rst]`` stay reStructuredText, which helps migrating old examples:
#
# .. math::
#
#    \nabla \cdot \mathbf{v} = 0
#
# * item one
# * item two

# %% tags=["remove-input"]
print("This output comes from a cell whose code is removed (remove-input).")

# %% [markdown]
# ## A long output
#
# Text outputs longer than `--max-output-lines` are truncated in the middle.

# %%
for i in range(600):
    print(f"step {i}")
