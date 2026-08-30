"""
sort by keys performance benchmarks
====================================

This example benchmarks the sort by keys (general, non power-of-2 length) performance for
the different algorithms available in Shamrock, as well as the sort_by_key_pow2_len
(power-of-2 length only) performance in a second figure.
"""

# sphinx_gallery_multi_image = "single"

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# List current implementation
if not shamrock.algs.is_impl_set_sort_by_keys():
    shamrock.algs.autoselect_impl_sort_by_keys()

current_impl = shamrock.algs.get_current_impl_sort_by_keys()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_sort_by_keys()

print(all_default_impls)

# %%
# Pretty printer


_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _sorted_run_mask(values):
    """Flag indices that belong to a non-decreasing run of length >= 2."""
    mask = [False for i in range(len(values))]

    for i, v in enumerate(values):
        if i > 0:
            if values[i - 1] < v:
                mask[i] = True
        else:
            mask[i] = True
    return mask


def print_key_val_table(keys, vals, labels=("i", "key", "val")):
    """Pretty-print indices/keys/values as an aligned table, coloring any
    sorted (non-decreasing) run of >= 2 consecutive keys in green.

    i   |  0   1   2  ...
    ----------------------
    key |  ?   ?   ?  ...
    val |  ?   ?   ?  ...
    """
    idx_label, key_label, val_label = labels
    rows = [idx_label, key_label, val_label]
    columns = [list(range(len(keys))), keys, vals]

    label_width = max(len(row) for row in rows)
    col_width = max(len(str(v)) for col in columns for v in col)

    def cell(v, highlight=None):
        text = f"{v!s:>{col_width}}"
        if highlight is None:
            return text
        return f"{_GREEN}{text}{_RESET}" if highlight else f"{_RED}{text}{_RESET}"

    def fmt_row(label, values, mask=None):
        cells = "  ".join(cell(v, mask[i] if mask else None) for i, v in enumerate(values))
        return f"{label:<{label_width}} | {cells}"

    idx_line = fmt_row(idx_label, columns[0])
    key_line = fmt_row(key_label, columns[1], _sorted_run_mask(keys))
    val_line = fmt_row(val_label, columns[2])

    print("-" * len(idx_line))
    print(idx_line)
    print("-" * len(idx_line))
    print(key_line)
    print(val_line)


def to_buf(lst):
    buf = shamrock.backends.DeviceBuffer_u32()
    buf.resize(len(lst))
    buf.copy_from_stdvec(lst)
    return buf


# %%
# Dataset
N = 100
key_init = [i for i in range(N)][::-1]
val_init = [i for i in range(N)]

print("Initial state:")
print_key_val_table(key_init, val_init)


# %%
# Switch impl
shamrock.algs.set_impl_sort_by_keys('{"implementation":"modern_gpu_mergesort","parameters":{}}')

# %%
# End state
buffer_key = to_buf(key_init)
buffer_val = to_buf(val_init)

shamrock.algs.sort_by_keys(buffer_key, buffer_val, N)

print("End state:")
print_key_val_table(buffer_key.copy_to_stdvec(), buffer_val.copy_to_stdvec())

# %%
# Expected state
buffer_key = to_buf(key_init)
buffer_val = to_buf(val_init)

shamrock.algs.autoselect_impl_sort_by_keys()
shamrock.algs.sort_by_keys(buffer_key, buffer_val, N)

print("Expected state:")
print_key_val_table(buffer_key.copy_to_stdvec(), buffer_val.copy_to_stdvec())
