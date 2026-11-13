# %% [markdown]
# # A script that is shown but not executed
#
# Files whose name does not start with `run_` are rendered without running them,
# like `filename_pattern="/run_"` in the current sphinx-gallery setup.

# %%
def expensive_setup():
    raise RuntimeError("this would fail if it were executed")


expensive_setup()
